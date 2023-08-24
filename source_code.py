!wget http://cb.lk/covid_19
!unzip covid_19
import keras
from keras.models import *
from keras.layers import *
from keras.preprocessing import image
import PIL
import os
os.listdir('/content/CovidDataset')
os.listdir('/content/CovidDataset/Train/Covid')
len(os.listdir('/content/CovidDataset/Train/Covid'))
len(os.listdir('/content/CovidDataset/Val/Covid'))

PIL.Image.open('/content/CovidDataset/Train/Covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg')
os.listdir('/content/CovidDataset/Train/Covid')[10]
os.listdir('/content/CovidDataset')[0]
dir='/content/CovidDataset/Train/Covid'
img_dir=os.path.join(dir,'covid-19-pneumonia-22-day2-pa.png')
#img_dir=os.path.join(dir,os.listdir('/content/CovidDataset/Train/Covid')[10])
PIL.Image.open(os.path.join(dir,os.listdir('/content/CovidDataset/Train/Covid')[10]))#PIL.Image.open(img_dir)
from keras.preprocessing.image import ImageDataGenerator
Train_datgen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
Test_datgen=ImageDataGenerator(rescale=1./255)
Train_set=Train_datgen.flow_from_directory(r'/content/CovidDataset/Train',target_size=(224,224),
                                           batch_size=32,class_mode='binary')
Test_set=Test_datgen.flow_from_directory(r'/content/CovidDataset/Val',target_size=(224,224),
                                           batch_size=32,class_mode='binary')
model = Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)))#222*222*32



model.add(Conv2D(64,(3,3),activation="relu"))#220*220*64
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1,activation="sigmoid"))

model.compile(loss=keras.losses.binary_crossentropy,optimizer="adam",metrics=['accuracy'])
model.summary()
result=model.fit_generator(Train_set,steps_per_epoch=2,epochs=10,validation_data=Test_set)
result.history.keys()
import matplotlib.pyplot as plt

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.legend(['Training_loss','Validation_loss'])
plt.xlabel('epoch')
plt.ylabel('Loss')

plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.legend(['Training_accuracy','Validation_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
import os.path
if os.path.isfile('models/covid_19.h5') is False:
  model.save('models/covid_19.h5')
from tensorflow.keras.models import load_model
new_model=load_model('models/covid_19.h5')
#new_model.summary()
new_model.get_weights()
new_model.optimizer
pred=new_model.evaluate(Test_set)
pred
from skimage import io
from skimage.transform import resize

img=io.imread(r'enter ur path here to test the model')
img=resize(img,(1,224,224,3))
img_pred=new_model.predict(img)
img_pred
if img_pred[0][0] > 0.5 :
  print('Positive')
else :
  print('Negative')