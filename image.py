# tf.keras.datasets
# from tensorflow.keras.datasets import mnist,cifar10,cifar100,fashion_mnist
# import numpy as np

# (train_X,train_y),(test_X,test_y)=cifar10.load_data()
# (train_X,train_y),(test_X,test_y)=cifar100.load_data()
# (train_X,train_y),(test_X,test_y)=imdb.load_data()

# (train_X,train_y),(test_X,test_y)=fashion_mnist.load_data()
# print(train_X.shape)
# print(train_y.shape)
# print(train_y) #for checking  data
# print(np.unique(train_y))


#RGB :-3D IMAGE
# GRACE-SCALE IMAGE:-2D IMAGE(WHITE=255 FREQUENCY AND BLACK COLOR=0 FREQUENCY) IN BTWN 0-255 THERE ARE DIFFERENT FORMS OF BALCK AND GRAY.
# 1PIXEL HAVE 3 VALUES AS IT AHS 3 COLORS WHICH IS BLACK,WHITE,GRAY

# to read image
# import matplotlib.pyplot as plt
# for x in range(25):
#     plt.subplot(5,5,x+1)
#     plt.imshow(train_X[x])
#     print(train_y[x])
# plt.show()

# # RESHAPE THE DATA OF TRAIN_x
from tensorflow.keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import numpy as np

(train_X,train_y),(test_X,test_y)=mnist.load_data()
print(train_X.shape)
train_X=train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2],1)
test_X=test_X.reshape(test_X.shape[0],test_X.shape[1],test_X.shape[2],1)

print(train_X.shape)

train_X=train_X.astype("float32")
test_X=test_X.astype("float32")

train_X=train_X/255
test_X=test_X/255

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

model=Sequential()
in_shape=train_X.shape[1:]
model.add(Conv2D(2,(2,2),activation="relu",input_shape=in_shape))
model.add(MaxPool2D((3,3)))
model.add(Conv2D(2,(2,2),activation="relu",input_shape=in_shape))
model.add(MaxPool2D((3,3)))
# model.add(Conv2D(2,(2,2),activation="relu",input_shape=in_shape))
# model.add(MaxPool2D((3,3)))

model.add(Flatten())
model.add(Dense(10,activation="relu"))
model.add(Dense(60,activation="relu"))
model.add(Dense(10,activation="sigmoid"))
print(model)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_X,train_y,batch_size=32,epochs=5,verbose=0)      #we  take less epochs to get output fast but as epochs value increase the time of output increase itself 
loss,accuracy=model.evaluate(test_X,test_y)

print("Accuracy:",accuracy)
model.save('image.h5')


from tensorflow.keras.models import load_model
model=load_model("image.h5")
print(model)
image=test_X[0]
y_pred=model.predict(np.asarray([image]))
# y_pred=model.predict(np.asarray(test_X[0]))

print("Predicted digit:",np.argmax(y_pred))
print("actual digit:",test_y[0])


# print(y_pred)
# print(y_test[0])


#############06-05
# model=models.Sequential()
# in_shape=trainimages.shape[1:]
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))

# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10))
# model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# test_loss,test_acc= model.evaluate(test_images,test_labels,verbose=2)
# print(test_acc)