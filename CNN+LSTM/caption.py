#import requirements

import glob
#from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import nltk
import random
#from scipy.misc import imread
#import spacy
#import wmd
from keras.models import Model
#from nltk.translate.bleu_score import sentence_bleu

class Caption():
    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.word2idx = None
        self.idx2word = None
        self.d = None
        self.train_d = None
        self.lengthImg = None
        self.model_new = None
        self.processtrainData()
        #self.total_samples = None
        #self.encoded_images = pickle.load( open( "/home/manish.singhal/Image-Captioning-master/encoded_images.p", "rb" ) )
        self.variable_initializer()
        
    def split_data(self,l,img,lengthImages):
        temp = []
        for i in img:
            if i[lengthImages:] in l:
                temp.append(i)
        return temp

    def preprocess_input(self,x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def preprocess(self,image_path):
        #print (image_path)
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = self.preprocess_input(x)
        return x

    def encode(self,image):

        #print (image)
        image = self.preprocess(image)
        temp_enc = self.model_new.predict(image)
        temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
        return temp_enc

    def processtrainData(self):
   
        token = '../processed_files/Flickr8k.token.txt'
   
        captions = open(token, 'r').read().strip().split('\n')
   
        d = {}
        for i, row in enumerate(captions):
            row = row.split('\t')
            row[0] = row[0][:len(row[0])-2]
            if row[0] in d:
                d[row[0]].append(row[1])
            else:
                d[row[0]] = [row[1]]
        self.d = d
        images = 'Flicker8k_Dataset/'
    
        img = glob.glob(images+'*.jpg')
    
        print(len(img))
        #train_images_file = 'Flickr_8k.trainImages.txt'
        #train_images_file = 'trainImages.txt'
        train_images_file = '../processed_files/Flickr_8k.trainImages.txt'
        train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
        self.lengthImg = len(images)
        train_img = self.split_data(train_images,img,self.lengthImg)
        train_dict = {}
        for i in train_img:
            if i[len(images):] in d:
                train_dict[i] = d[i[len(images):]]
        self.train_d = train_dict
    
    def variable_initializer(self):
        train_d = self.train_d
        unique = pickle.load(open('../processed_files/unique.p', 'rb'))
        caps = []
        for key, val in train_d.items():
            for i in val:
                caps.append('<start> ' + i + ' <end>')
 
        words = [i.split() for i in caps]
        #unique = []
        #for i in words:
           # unique.extend(i)
        #unique = list(set(unique))
        max_len = 0
        for c in caps:
            c = c.split()
            if len(c) > max_len:
                max_len = len(c)
        self.max_cap_len = max_len
        self.vocab_size = len(unique)
        #self.d = dic
        self.word2idx = {val:index for index, val in enumerate(unique)}
        self.idx2word = {index:val for index, val in enumerate(unique)}

        model = InceptionV3(weights='imagenet')
        new_input = model.input
        hidden_layer = model.layers[-2].output
        self.model_new = Model(new_input, hidden_layer)
        
    def create_model(self):
        #model = InceptionV3(weights='imagenet')
        embedding_size = 300
        
        image_model = Sequential([Dense(embedding_size, input_shape=(2048,), activation='relu'),RepeatVector(self.max_cap_len)])

        caption_model = Sequential([ Embedding(self.vocab_size, embedding_size, input_length=self.max_cap_len),LSTM(256, return_sequences=True), TimeDistributed(Dense(300))])

        final_model = Sequential([ Merge([image_model, caption_model], mode='concat', concat_axis=1),Bidirectional(LSTM(256, return_sequences=False)), Dense(self.vocab_size),Activation('softmax')])

        final_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        final_model.summary()
        #final_model.load_weights('weights-improvement_epoch50_plateau-49.hdf5')
        return final_model
    

