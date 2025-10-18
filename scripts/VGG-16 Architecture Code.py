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
zoom_range=0.15,
width_shift_range=0.15 ,
height_shift_range=0.15 ,
shear_range=0.10 ,
horizontal_flip=True,
brightness_range=[0.8,1.2],
channel_shift_range=0.15,
fill_mode = "nearest")
test_datagen=ImageDataGenerator(rescale = 1./255)
x_train = train_datagen.flow_from_directory('/content/drive/MyDrive/The lung cancer dataset/trainset',target_size=(224,224),batch_size=32,class_mode='binary')
x_test=test_datagen.flow_from_directory('/content/drive/MyDrive/The lung cancer dataset/testset',target_size=(224,224),batch_size=32,class_mode = 'binary')
print(x_train.class_indices)
from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape = (224,224,3),# Shape o f our images
include_top = False, # Leave ou t t h e l a s t f u l l y c onnec te d l a y e r
weights = 'imagenet')
for layer in base_model.layers:
  layer.trainable = False
x = layers.Flatten()(base_model.output)
x = layers.Dense(512,activation = 'relu')(x)
# Add a d r o p o u t r a t e o f 0 . 5
x = layers.Dropout(0.5)(x)
# Add a f i n a l s igm o i d l a y e r f o r c l a s s i f i c a t i o n
x = layers.Dense(1,activation= 'sigmoid')(x)
model = tf.keras.models.Model(base_model.input,x)
model.compile(loss = "binary_crossentropy",optimizer= 'adam',metrics=['accuracy'])
TRAIN_COUNT = len(x_train.filepaths)
TEST_COUNT = len(x_test.filepaths)
TRAIN_STEPS_PER_EPOCH = round(TRAIN_COUNT/32 )
VAL_STEPS_PER_EPOCH = round(TEST_COUNT/32 )
print( f"TRAIN_STEPS_PER_EPOCH: {TRAIN_STEPS_PER_EPOCH}")
print ( f"VAL_STEPS_PER_EPOCH: {VAL_STEPS_PER_EPOCH}")
from tensorflow.python.keras.callbacks import EarlyStopping , ModelCheckpoint
EARLY_STOP_PATIENCE=5
cb_early_stopper = EarlyStopping(monitor='val_loss',patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath='best.hdf5',
monitor='val_loss',save_best_only=True , mode='auto')
callbacks =[cb_checkpointer,cb_early_stopper]
history = model.fit(x_train,
steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
epochs = 20,validation_data = x_test,
validation_steps = VAL_STEPS_PER_EPOCH)
from sklearn.metrics import confusion_matrix , classification_report
x_pred=model.predict(x_test)
x_p=np.argmax(x_pred,axis =1)
x_true=np.argmax (x_test,axis =1)
print(confusion_matrix(x_true,x_p))
from sklearn.metrics import accuracy_score
x_pred=model.predict(x_test)
print(accuracy_score(x_test,x_pred))
import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(x_test,x_pred),annot=True)
keys = history.history.keys()
print(keys)
plt.figure(1,fig_size = (15,8))
plt.subplot(221)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
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
plt.legend(['train', 'valid'])
plt.show()
for key in keys:
  plt.figure(figsize = (15,2))
  plt.plot(history.history[key])
  plt.title(key)
  plt.ylabel(key)
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'])
  plt.show()
model.save("best.h5")
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
model=load_model('/content/best.hdf5')
from keras.preprocessing import image
import cv2
import numpy as np
import PIL
from PIL import Image
img=image.load_img('/content/drive/MyDrive/The lung cancer dataset/testset/non cancer/Normal case (387).jpg',targetsize=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis =0)
x.shape
pred= model.predict(x)
pred
class_names=["Cancer","NonCancer"]
prediction=class_names[int(pred[0][0]) ]
print(pred[0][0])
print(prediction)