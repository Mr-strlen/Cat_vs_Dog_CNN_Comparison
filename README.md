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
  
训练过程如下,共设置了30个epoch:  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/base_cnn_process.png)  
Acc和Loss的结果如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/base_cnn_acc.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/base_cnn_loss.png)  
可见最后测试集正确率大概在70%，并且由于每个epoch内部图片存在差异，所以在某些eopch的时候出现一些反弹。  
  
这里并不是过拟合，因为之后增加网络层数至三层（数据并没有全部使用，也没有扩充），loss会继续下降，如下图：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/base_cnn_loss2.png)  
当然避免这种浮动较大的好办法就是扩充数据，这也将在之后的参数调整中看到效果。  


### AlexNet网络
这个AlexNet就不介绍了，直接放一张结构图：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/alexnet_model.png)  
  
Keras中没有直接的AlexNet网络，所以就得按照上面一层层加上去。最后网络的结构如下图：
```python
# AlexNet
model = Sequential()
#第一段
model.add(Conv2D(filters=96, kernel_size=(11,11),
                 strides=(4,4), padding='valid',
                 input_shape=(resize,resize,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), 
                       strides=(2,2), 
                       padding='valid'))
#第二段
model.add(Conv2D(filters=256, kernel_size=(5,5), 
                 strides=(1,1), padding='same', 
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), 
                       strides=(2,2), 
                       padding='valid'))
#第三段
model.add(Conv2D(filters=384, kernel_size=(3,3), 
                 strides=(1,1), padding='same', 
                 activation='relu'))
model.add(Conv2D(filters=384, kernel_size=(3,3), 
                 strides=(1,1), padding='same', 
                 activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), 
                 strides=(1,1), padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), 
                       strides=(2,2), padding='valid'))
#第四段
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))
```
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/alexnet_structure.png)  
整体网络还是非常庞大的，在三十次epoch后，准确率约为70%左右。  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/alexnet_process.png)  
具体的ACC曲线和Loss曲线如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/alexnet_acc.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/alexnet_loss.png)  
虽然在训练集上，模型越来越好，但是在推广上并没有太大的优势。当然这个和我自己写的网络有关，这里的AlexNet网络代码中，没有进行图像变换的数据增强，单次epoch的数据量也仅为4000张图片，这里就不做更多改进了，也能和之后的其他模型对比看出差异。


### VGG16网络
这个VGG16就不介绍了，直接放一张结构图：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/vgg16_model.png)  
这里我也是使用了Keras自带的VGG16网络，值得提醒的一点是，这里的VGG16最后的softmax全连接层输出结果为1 * 1000，但是我们这里是个二分类问题，所以，需要我们手动再加上一层全连接层改变输出结果，具体可以看公开的代码 (https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py)  
  
所以最后网络定义如下：
```python
#VGG16网络
conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(150, 150, 3))
#构建网络
model = models.Sequential() 
model.add(conv_base) 
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-5), 
              metrics=['acc'])
```
也是采用了sigmoid函数作为分类结果，使用二元交叉熵函数，利用RMSprop方法优化。结构图如下:  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/vgg16_structure.png)  
  
有关图像变换的部分，参考了一下别人的设定，设置如下：
```python
train_datagen = ImageDataGenerator(
    rotation_range=15,#数据提升时图片随机转动的角度。随机选择图片的角度
    rescale=1./255,#值将在执行其他处理前乘到整个图像上
    shear_range=0.1,#剪切强度（逆时针方向的剪切变换角度）
    zoom_range=0.2,#随机缩放的幅度。用来进行随机的放大。
    horizontal_flip=True,#进行随机水平翻转。
    width_shift_range=0.1,#随机水平偏移的幅度
    height_shift_range=0.1#随机竖直偏移的幅度
    #height_shift_range和width_shift_range是用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例
)
```
训练过程：
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/alexnet_process.png)  
  
设置验证集大小为 20% ，训练集是20000张图，验证集5000张图。单次epoch大概在20s左右，acc逐渐上升，最后达到97%，Loss降到0.1以下，具体的Acc和loss图如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/vgg16_acc.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/vgg16_loss.png)  


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
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-Comparison/blob/master/Images/Keras_application.png)  
这次报告中的VGG16,ResNet,Xception就是直接掉包使用，非常的方便。  
  
最后贴一下我的运行环境，我的台式机运了回来，有了显卡，所以性能还是很不错的。 
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-Comparison/blob/master/Images/computer_base.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-Comparison/blob/master/Images/GPU_base.png)
