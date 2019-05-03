#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('env', 'KERAS_BACKEND=tensorflow')
from keras.models import Sequential ,Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.optimizers import SGD
from keras.layers import LeakyReLU


# In[2]:


def construct_Conv2D(f_num,f_size,stride=(1,1)):
    return model.add(Conv2D(f_num,f_size,strides=stride,padding="same"))


# In[5]:


IMAGE_H = 720
IMAGE_W = 1280
GRID_H , GRID_W= 20 , 20
input_img=( IMAGE_H , IMAGE_W ,3)


# In[10]:


model=Sequential()


# In[11]:


##Layer 1
model.add(Conv2D(64,(18,32),strides=(9,16),padding="same",input_shape=input_img))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

## Layer 2
construct_Conv2D(64,(3,3))
model.add(LeakyReLU(alpha=0.1))
##model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

## Layer 3
construct_Conv2D(128,(1,1))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(256,(3,3))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(256,(1,1))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

##model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

## Layer 4
construct_Conv2D(256,(1,1))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(1,1))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

##model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

## Layer 5
construct_Conv2D(512,(1,1))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))## strides(2,2)
model.add(LeakyReLU(alpha=0.1))

## Layer 6
construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

construct_Conv2D(512,(3,3))
model.add(LeakyReLU(alpha=0.1))

## Last two layer
model.add(Conv2D(8,(1,1),padding="same"))
model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())
model.add(Dense(2880,input_dim=12800))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.add(Reshape((5,18,32)))


# In[12]:


model.summary()


# In[ ]:




