#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[4]:


import csv
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# # Read the train and test set 

# In[5]:


train_dataset='.digits/train.csv'
test_dataset='.digits/test.csv'
#read csv files
train_df=pd.read_csv(train_dataset)
test_df=pd.read_csv(test_dataset)

train_no_of_values,train_no_of_features=(train_df.shape)
#train set for validation=train_no_of_values/5
test_no_of_values,test_no_of_features=(test_df.shape)


# # Identify the dependent and independent variables

# In[6]:


y_train_df=train_df.iloc[:,0]
x_train_df=train_df.iloc[:,1:]


# # Create numpy arrays

# In[7]:


list=[]
for i in range(int(4*train_no_of_values/5)):
    list.append(y_train_df[i])
y_train=np.array(list)
print(y_train.shape)


# In[8]:


list=[]
for i in range(int(4*train_no_of_values/5),train_no_of_values):
    list.append(y_train_df[i])
y_validate=np.array(list)
print(y_validate.shape)


# In[9]:


list=[]
for k in range(int(4*train_no_of_values/5)):
    list2=[]
    for i in range(28):
        list1=[]
        for j in range(28):
            list1.append(x_train_df.iloc[k,i * 28 + j])
        list2.append(list1)
    list.append(list2)
x_train=np.array(list)


# In[10]:


list=[]
for k in range(int(4*train_no_of_values/5),train_no_of_values):
    list2=[]
    for i in range(28):
        list1=[]
        for j in range(28):
            list1.append(x_train_df.iloc[k,i * 28 + j])
        list2.append(list1)
    list.append(list2)
x_validate=np.array(list)


# In[11]:


list=[]
for k in range(test_no_of_values):
    list2=[]
    for i in range(28):
        list1=[]
        for j in range(28):
            list1.append(test_df.iloc[k,i * 28 + j])
        list2.append(list1)
    list.append(list2)
x_test=np.array(list)
print(x_test.shape) 


# # Preprocess the data

# In[12]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validate = x_validate.reshape(x_validate.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=None)#convert labels to categorical features i.e.0/1
y_validate = keras.utils.to_categorical(y_validate, num_classes=None)
x_train = x_train.astype('float32')
x_validate = x_validate.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_validate /= 255
x_test /= 255


# In[13]:


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_validate shape:', y_validate.shape)


# # Create the model

# In[14]:


num_classes = 10
epochs = 10
model = Sequential()
#input shape is needed only for first layer
#activation is 'relu' because it does not need normalisation
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))#32 filters each of size3*3
model.add(MaxPooling2D(pool_size=(2, 2)))#to create MaxPooling layer
model.add(Dropout(0.25))
model.add(Flatten())#to flatten it up into neural network 
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# # Fit the model

# In[15]:


model.fit(x_train, y_train,epochs=epochs,validation_data=(x_validate, y_validate))


# # Find the value accuracy and loss

# In[16]:


score=model.evaluate(x_validate,y_validate,batch_size=10)
print(score)


# # Save the model

# In[57]:


from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 


# # Make the predictions

# In[2]:


x_test_predict=x_test.copy()#make a copy of x_test
x_test_predict = x_test_predict.reshape(x_test_predict.shape[0], 28, 28, 1)#reshape it


# In[55]:


jm_list=[]
for i in range(test_no_of_values):
    #print(i)
    result_new = model.predict([x_test_predict])[i]
    number=np.argmax(result_new)
   # accuracy=max(result_new)
    jm_list.append(number)


# # Check manually that the results match using the image ( just for confirmation) # Optional

# In[1]:


x=int(input('Enter index of image you wish to view'))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(x_test[x-1])


# # Save results in file

# In[3]:


row_list = [["ImageId", "Label"]]
for i in range(test_no_of_values):
    temp_list=[]
    temp_list.append(i+1)
    temp_list.append(jm_list[i])
    row_list.append(temp_list)


# In[21]:


with open('result_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)


# In[ ]:




