# Cat_vs_Dog-CNN-Comparison
目录
---
* [实验内容](#实验内容)
    * [数据集简介](#数据集简介)
    * [CNN网络构建](#CNN网络构建)
    * [AlexNet网络](#AlexNet网络)
    * [VGG16网络](#VGG16网络)
* [实验结果和分析](#实验结果和分析)
    * [CNN参数调整](#CNN参数调整)
    * [ResNet](#ResNet)
    * [Xception](#Xception)
    * [ResNet50+Xception+Inception V3](#ResNet50+Xception+InceptionV3)
* [各种方法比较](##各种方法比较)
* [个人总结](##个人总结)
***
我选择的是16年Kaggle的猫狗大战数据集。之所以选择这个数据集，是因为自己一直很想把不同的网络方法，AlexNet，VGG16，ResNet50等等放在一起，做一个比较（就那种各种方法结果放在一张图里的，特别酷炫）。</br>
所以这次涉及到的代码量就比较多，之后的论述中就只粘贴核心代码部分。如果是我直接使用别人现成的模型或者代码，我也会注上网址。毕竟相比很多大牛，我还算一个门都没入的小白。
## 实验内容
### 数据集简介
数据集来源于此[Kaggle官网](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)</br>
训练集有25000张，猫狗各占一半。测试集12500张，没有标定是猫还是狗。</br>
有大佬对数据本身做了一个分布的调查，分布结果如下(https://www.jianshu.com/p/1bc2abe88388)</br>
训练集中图片的尺寸散点分布图：</br>
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/scatter_diagram_train_dataset.png)</br>
测试集中图片的尺寸散点分布图：</br>
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/scatter_diagram_test_dataset.png)</br>
可以发现图像的大小存在着差异性，这就需要我们上来对图像进行一定的预处理。  
  
还有大佬发现，存在部分过于难识别的图像，通过手动删除的方式进行预处理，这里我觉得很有意思，也贴在这，但是之后的实验并没有进行数据的剔除。</br>
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/image_movement.png)
### CNN网络构建
这里使用了Keras模型构建（正好之前用过）具体代码如下</br>
```python
#开始序列模型
model = models.Sequential()
#第一层卷积池化
model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#第二层卷积池化
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#第三层全连接层
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```
可以看到最后设定了loss使用交叉熵，使用RMSProp算法优化过程（优化算法包括GD，SGD，Momentum，RMSProp和Adam算法，其中基于均方根传播，而Adam算法接近物理中动量来累积梯度）  
  
model.summary()会输出模型各层的参数状况，可以看到使用了两层卷积池化网络最后全连接层输出：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/base_cnn_structure.png)  
  
这里有两个有趣的点，一个是parma，这是Keras特有的计算网络大小的参数，说明的是每层神经元权重的个数，计算公式如下：  
***Param = (输入数据维度 + 1) * 神经元个数***  
***之所以要+1，是考虑到每个神经元都有一个Bias***
  
那对于卷积神经网络即有：  
***(卷积核长度 * 卷积核宽度 * 通道数 + 1) * 卷积核个数***  
所以第一个CONV层，Conv2D(32, (3, 3), input_shape=(150, 150, 3))，Param = (3 * 3 * 3 + 1) * 32 = 896  
  
另一个是最后全连接的层的选择，之前我们都使用的是softmax函数，这里却选择了sigmoid函数，这里有个小知识点，就是在二分类的时候，sigmoid函数和softmax函数完全等价，推导可以看这个(https://www.zhihu.com/question/295247085)  
sigmoid和softmax是神经网络输出层使用的激活函数，分别用于两类判别和多类判别，binary cross-entropy和categorical cross-entropy是相对应的损失函数。对应的激活函数和损失函数相匹配，可以使得error propagation的时候，每个输出神经元的“误差”（损失函数对输入的导数）恰等于其输出与ground truth之差。（详见Pattern Recognition and Machine Learning一书5.3章）  


### AlexNet网络
### VGG16网络
## 实验结果和分析
### CNN参数调整
### ResNet
### Xception
### ResNet50+Xception+InceptionV3
## 各种方法比较
终于到我觉得很酷炫的环节了，前面我们一共有6种CNN模型，这里我们将6种模型放在一起比较：  
从网络规模上来看，AlexNet作为早期CNN网络，比较繁重，但是正是大量的神经元，才具有跨时代的作用。之后的网络进行结构优化，减少网络规模，提高效率。  单位epoch用时的数据我也贴上去，但是参考意义不大，一方面是batch_size并不完全相同，并且后面的复杂网络都是用训练好的模型，就做个参考吧。  
下面是在测试集上Acc曲线和Loss曲线的比较，这里都是30次epoch，所以还是具有比较价值的（部分和之前给的图不一样，是因为后来重新算补的数据，不过趋势是没有变化的）：  
基本上越是后来出现的网络，效果越好，非常符合CNN网络的发展趋势。其中ResNet因为用了50层，所以出现了过拟合，AlexNet也是出现了这样的情况，这个算是情理之中吧。  
## 个人总结
这次算是满足了我个人小小的心愿。以前看到各种模型放在一起比较非常的酷炫，现在我也可以照这样子自己做一个，还是很有成就感的。  
然后Keras.application这个包里，有很多训练好可以直接用网络：
这次报告中的VGG16,ResNet,Xception就是直接掉包使用，非常的方便。  
最后贴一下我的运行环境，我的台式机运了回来，有了显卡，所以性能还是很强悍的。  
