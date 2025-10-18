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
train_datagen = ImageDataGenerator (rescale = 1./255, rotation_range=8,
                                    zoom_range =0.15 ,
                                    width_shift_range=0.15 ,
                                    height_shift_range=0.15 ,
                                    shear_range=0.10,
                                    horizontal_flip=True,
                                    brightness_range=[0.8,1.2],
                                    channel_shift_range=0.15,
                                    fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1./255)
x_train = train_datagen.flow_from_directory('/content/drive/MyDrive/The lung cancer dataset/trainset',target_size = (299,299) ,batch_size=32,class_mode = 'binary')
x_test = test_datagen.flow_from_directory('/content/drive/MyDrive/The lung cancer dataset/testset',
target_size = (299,299),batch_size=32,
class_mode = 'binary', shuffle=False)
print(x_train.class_indices)
from tensorflow.keras.applications import InceptionV3
base_model=InceptionV3(input_shape = (299,299,3) ,
include_top=False,
input_tensor=None ,
weights="imagenet",)
for layer in base_model.layers:
  layer.trainable = False
x=layers.Flatten()(base_model.output)
x=layers.Dense(512,activation= 'relu')(x)
x=layers.Dropout(0.5)(x)
# o/p layer for classification
x=layers.Dense(1,activation='sigmoid')(x)
model = tf.keras.models.Model(base_model.input,x)
model.compile(loss= "binary_crossentropy",
optimizer = 'adam',
metrics =['accuracy'])
TRAIN_COUNT = len(x_train.filepaths)
TEST_COUNT = len(x_test.filepaths)
TRAIN_STEPS_PER_EPOCH = round(TRAIN_COUNT/32)
VAL_STEPS_PER_EPOCH = round(TEST_COUNT/32)
print( f"TRAIN_STEPS_PER_EPOCH: {TRAIN_STEPS_PER_EPOCH}")
print( f"VAL_STEPS_PER_EPOCH: {VAL_STEPS_PER_EPOCH}")
history = model.fit_generator(x_train,
steps_per_epoch =TRAIN_STEPS_PER_EPOCH,
epochs =40,validation_data = x_test,
validation_steps = VAL_STEPS_PER_EPOCH)
model.save( "./AnjalInceptv3.h5")
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
threshold = 0.5
y_pred_threshold = [(1 if val>threshold else 0) for val in y_pred]
y_true = x_test.labels.tolist()
print(confusion_matrix (y_true,y_pred_threshold))
print(accuracy_score(y_true,y_pred_threshold))
sns.heatmap(confusion_matrix(y_true,y_pred_threshold),annot=True)
keys=history.history.keys()
print(keys)
plt.figure(1, figsize=(15,8))
plt.subplot(221)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'valid'])
plt.subplot(222)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','valid'])
plt.show()
for key in keys:
  plt.figure(figsize= (15,2))
  plt.plot(history.history[key])
  plt.title(key)
  plt.ylabel(key)
  plt.xlabel('epoch')
  plt.legend(['train','valid'])
  plt.show()
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.models import load_model
model=load_model('/content/AnjalInceptv3.h5')
from keras.preprocessing import image
import cv2
import numpy as np
import PIL
from PIL import Image
img=image.load_img('/content/drive/MyDrive/The lung cancer dataset/testset/non cancer/Normal case (390).jpg',target_size=(299 ,299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis =0)
x . shape
pred = model.predict(x)
pred
class_names=["Cancer","NonCancer"]
prediction=class_names[int(pred[0][0])]
print(pred[0][0])
print(prediction)