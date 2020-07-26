# -*- coding: utf-8 -*-
import os

from keras import layers, optimizers, models
from keras.applications.xception import Xception,preprocess_input
from keras.layers import *    
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd 

#准备数据
src_path = r'../'
dst_path = r'../'
train_dir = os.path.join(dst_path, 'train')
validation_dir = os.path.join(dst_path, 'valid')
test_dir = os.path.join(dst_path, 'test')

class_name = ['cat', 'dog']

#构建网络
base_model = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
model.summary()

from keras.preprocessing.image import ImageDataGenerator

#数据生成
filenames = os.listdir("../train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df.category=df.category.astype('str')
df.dtypes

print(df.head())
df.category.value_counts()


from sklearn.model_selection import train_test_split
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

train_df['category'].value_counts()
validate_df['category'].value_counts()

train_datagen = ImageDataGenerator(
    rotation_range=15,#整数，数据提升时图片随机转动的角度。随机选择图片的角度
    rescale=1./255,#值将在执行其他处理前乘到整个图像上
    shear_range=0.1,#浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度。
    zoom_range=0.2,#浮点数或形如[lower,upper]的列表，随机缩放的幅度。用来进行随机的放大。
    horizontal_flip=True,#布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
    width_shift_range=0.1,#浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
    height_shift_range=0.1#浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度
    #height_shift_range和width_shift_range是用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../train/", 
    x_col='filename',
    y_col='category',
    target_size=(150,150),
    class_mode='binary',
    batch_size=15
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../train/", 
    x_col='filename',
    y_col='category',
    target_size=(150,150),
    class_mode='binary',
    batch_size=15
)

history = model.fit_generator( 
    train_generator, 
    steps_per_epoch=100, 
    epochs=30,
    validation_data=validation_generator, 
    validation_steps=50)

#acc plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
