import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from google.colab import drive
drive.mount('/content/drive')
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator (rescale = 1./255,
rotation_range=8,
zoom_range =0.15 ,
width_shift_range=0.15 ,
height_shift_range=0.15 ,
shear_range=0.10 ,
horizontal_flip=True ,
brightness_range=[0.8 ,1.2 ] ,
channel_shift_range =0.15 ,
fill_mode="nearest")
test_datagen = ImageDataGenerator (rescale = 1./255)
x_train = train_datagen.flow_from_directory('/content/drive/My Drive/The lung cancer dataset/trainset',
target_size=(224,224),batch_size=32,
class_mode = 'binary')
x_test = test_datagen.flow_from_directory('/content/drive/My Drive/The lung cancer dataset/testset',
target_size = (224,224) , batch_size = 32 ,
class_mode = 'binary', shuffle=False)
print (x_train.class_indices)
model = Sequential( )
model.add (Convolution2D(32,(3,3),input_shape=(224,224,3),
activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units = 512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128,activation='relu'))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",
optimizer='adam',
metrics=['accuracy'])
history=model.fit_generator(x_train ,steps_per_epoch =24,
epochs=30,validation_data=x_test , validation_steps=  5)
model.summary()
model.save("FINAL_CNN. h5")
keys=history.history.keys()
print(keys)
plt.figure(1, figsize=(15,8))
plt.subplot(221)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','valid'])
plt.subplot(222)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model_Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','valid'])
plt.show()
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
threshold=0.5
y_pred_threshold = [(1 if val>threshold else 0 ) for val in y_pred ]
y_true=x_test.labels.tolist()
print(confusion_matrix(y_true,y_pred_threshold))
print(accuracy_score(y_true,y_pred_threshold))
sns.heatmap (confusion_matrix(y_true ,y_pred_threshold),annot=True)
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
model=load_model('./FINAL_CNN. h5')
from keras.preprocessing import image
import cv2
import numpy as np
import PIL
from PIL import Image
img=image.load_img( '/content/drive/My Drive/The lung cancer dataset/testset/cancer/Malignant case (493).jpg',
color_mode= 'rgb' ,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis =0)
x.shape
pred= model.predict(x)
pred
class_names=["Cancer", "NonCancer"]
prediction=class_names[int(pred[0][0])]
print(pred[0][0])
print(prediction)