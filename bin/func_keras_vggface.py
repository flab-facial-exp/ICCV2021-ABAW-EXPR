#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

import keras
from keras.engine import  Model
from keras.layers import Input
from keras.preprocessing import image
from keras import utils
from keras_vggface.vggface import VGGFace
from keras.applications.resnet50 import preprocess_input
from IPython.display import display, Image

"""
# Based on VGG16 architecture -> old paper(2015)
vggface = VGGFace(model='vgg16') # or VGGFace() as default

# Based on RESNET50 architecture -> new paper(2017)
vggface = VGGFace(model='resnet50')

# Based on SENET50 architecture -> new paper(2017)
vggface = VGGFace(model='senet50')
"""


# In[ ]:


class vggface:
    def __init__(self):
        self.model = VGGFace(include_top=False, model='resnet50', 
                           input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

    def get_vggface_features(self, filepath):
        img = image.load_img(filepath, target_size=(224, 224))
    
        x = np.array(img)
        x = np.expand_dims(x, axis=0)

        # After this point you can use your model to predict.
        features = self.model.predict(x).ravel()

        return features


# In[ ]:




