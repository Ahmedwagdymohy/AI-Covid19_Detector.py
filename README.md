# About


This code is a project that uses deep learning to classify COVID-19 and non-COVID-19 X-ray images. It first downloads and extracts a dataset, then utilizes Keras to create and train a convolutional neural network (CNN) model. The model architecture consists of convolutional and pooling layers, with dropout regularization, followed by dense layers. After compiling and training the model, it visualizes the training process using loss and accuracy plots. The trained model is saved, loaded, and used to predict new X-ray images, determining whether they show signs of COVID-19 or not.

# Foobar

AI-Covid19_Detector.py

## Installation

Install the required library which is : 

```bash
import keras
from keras.models import *
from keras.layers import *
from keras.preprocessing import image
import PIL
import os
```

## Usage
To test the model on  anew img : 
```python
img=io.imread(r'enter ur path here to test the model')

```

