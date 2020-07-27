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
首先，增加了数据，原本训练集数据只使用了2600张，这里扩充到8600张。  
网上基本上自己写的CNN都用到了三层，所以这里也是将网络扩展到3层，网络结构如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/new_cnn_structure.png)  
  
加入图像的一些变换：  
```python
train_datagen = ImageDataGenerator(
    rescale=1. / 255,#值将在执行其他处理前乘到整个图像上
    shear_range=0.2,#剪切强度（逆时针方向的剪切变换角度）
    zoom_range=0.2,#随机缩放的幅度，用来进行随机的放大。
horizontal_flip=True)#随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
```
运行过程如下，每次epoch内的数据约为原来的三倍：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/new_cnn_process.png)  
  
首先是正确率：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/new_cnn_acc.png)  
  
随着数据量增大，网络的加深，图形的变换,测试集正确率达到了83%。有意思的是在前9次epoch内，正确率并没有发生变化，但是我们观察Loss图像就可以看到，模型本身是在收敛的：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/new_cnn_loss.png)  
Loss也是从0.7降到了0.36


### ResNet
残差网络大名鼎鼎，所以我也是找了一个ResNet来看看结果（网上那些代码都用不了，这个代码是我自己写的OTZ）  
下图为ResNet34的结构，对于一个"Plain Network普通网络"，把它变为ResNet的方法是加上所有的跳远连接(skip connections).每两层增加一个跳远连接构成一个残差块，ResNet在训练深度网络方面非常有效: 
  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/resnet_model.png)  
  
与之类似的，随着层数的不同，还有ResNet18,ResNet50,ResNet101。ResNet50网络层数较多，就不展示具体的网络安排了。同样为了保证输出结果为二分类，加了一层全连接层。  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/resnet_structure.png)
  
值得一提的是Pytorch中有封装良好的ResNet网络，我也进行了尝试(https://www.cnblogs.com/weiba180/p/12417073.html)  
但是由于版本问题，只能使用单线程去跑，效率很低，最关键的是，它居然把我显卡显存跑满了，太恐怖了，所以还是继续使用Keras进行操作。  
  
图像预处理工作和之前是类似的，就不再赘述。
  
然后这里为了简化计算过程，直接下载了现成的ResNet50的模型，下载地址(https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5)
  
中间计算过程如下，可见每个epoch还是需要一定的时间计算：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/resnet_process.png)  

Acc和Loss曲线为：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/resnet_acc.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/resnet_loss.png)  
  
这里出现了一个ResNet很容易出现的问题，就是过拟合（网上很多人都有这个问题）这里主要是网络用的太深了（但是Keras自带的只有ResNet50，也不想自己再去写新的网络了）而且只有25000的图片，换成ResNet18在测试集上应该会有更好的效果。


### Xception
简单介绍一下Xception, Inception V3，这也是我第一次听到。  
  
Inception是一个模型家族，从V1进化到V4。Inception V1就是GoogleNet, Inception V2/V3有了总体设计原则、分解尺寸较大的卷积核、辅助分类器的效用、高效的降维方法(通过低维度嵌入来完成空间聚合)。  
  
Xception 是 Google 继 Inception 后提出的对 Inception-v3 的另一种改进。作者认为，通道之间的相关性 与 空间相关性 最好要分开处理。采用 Separable Convolution（极致的 Inception 模块）来替换原来 Inception-v3中的卷积操作。  
  
总而言之，Xception 是由 Inception结构演变而来,借鉴了 Depthwise Convolution思想的架构, 同时使用了ResNet的思想，具体的结构如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/xception_model.png)  
  
Keras中也是直接提供了Xception的网络结构(https://www.cnblogs.com/zhengbiqing/p/12008482.html)  
结合之前的代码，很容易写出猫狗大战的demo，具体的网络结构层数很多，就不截图放上来了，但是可以看到总共有两千多万个params，网络十分复杂。  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/xception_structure.png)  
  
这里也是直接下载了训练好的模型（我的电脑不可能跑完这个模型的）  
计算过程如下，在使用已经训练好的模型下，单个epoch还需要长达30s的训练，可见网络规模之大：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/xception_process.png)  
最后的Acc和Loss图像，正确率达到了98%，loss也小于了0.05，而且这种复杂的网络完全没有出现过拟合的情况，可以说是非常优秀了：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/xception_acc.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/xception_loss.png)  


### ResNet50+Xception+InceptionV3
Xception的结果已经非常优秀了，但是我还是找到了更厉害的思路。这里就要介绍一个大神的思路了(https://zhuanlan.zhihu.com/p/25978105)  
号称能够在Kaggle平台上跑到20名左右的成绩，简单说一下他的思路。  
  
大神试了一下各种预训练的网络，发现排名都不行，那么一种有效的方法是综合各个不同的模型，从而得到不错的效果，所以他使用了ResNet50, Xception, Inception V3 这三个模型，结构模型如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/three_model.png) 
  
我也是把大神的代码拿了过来，自己跑了一下，结果如下：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/three_process.png) 
设置验证集大小为 20% ，也就是说训练集是20000张图，验证集是5000张图。为了和前面的epoch数量一致，我这里也是30次。  
非常厉害，从第二次epoch之后每次训练只需要1s，而且正确率达到了恐怖的99.6%，loss连0.01都不到，可见模型整合的厉害之处。  


## 各种方法比较
终于到我觉得很酷炫的环节了，前面我们一共有6种CNN模型，这里我们将6种模型放在一起比较：  
  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/compare_total.png)  
从网络规模上来看，AlexNet作为早期CNN网络，比较繁重，但是正是大量的神经元，才具有跨时代的作用。之后的网络进行结构优化，减少网络规模，提高效率。   
  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/compare_epoch.png)  
单位epoch用时的数据我也贴上去，但是参考意义不大，一方面是batch_size并不完全相同，并且后面的复杂网络都是用训练好的模型，就做个参考吧。  
  
下面是在测试集上Acc曲线和Loss曲线的比较，这里都是30次epoch，所以还是具有比较价值的（部分和之前给的图不一样，是因为后来重新算补的数据，不过趋势是没有变化的）：  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/compare_acc.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-compare/blob/master/Images/compare_loss.png)  
  
基本上越是后来出现的网络，效果越好，非常符合CNN网络的发展趋势。其中ResNet因为用了50层，所以出现了过拟合，AlexNet也是出现了这样的情况，这个算是情理之中吧。  


## 个人总结
这次做的很简单很简单，不过也算是满足了我个人小小的心愿。以前看到各种模型放在一起比较非常的酷炫，现在我也可以照这样子自己做一个，还是有点成就感的。  
  
然后Keras.application这个包里，有很多训练好可以直接用网络：
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-Comparison/blob/master/Images/Keras_application.png)  
这次报告中的VGG16,ResNet,Xception就是直接掉包使用，非常的方便。  
  
最后贴一下我的运行环境，我的台式机运了回来，有了显卡，所以性能还是很不错的。 
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-Comparison/blob/master/Images/computer_base.png)  
![image](https://github.com/Mr-strlen/Cat_vs_Dog-CNN-Comparison/blob/master/Images/GPU_base.png)
