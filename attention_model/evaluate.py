import os, sys
import pickle 
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np
import glob
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd

from keras.models import Model, load_model
from model import image_caption_model
from beamsearch import beamsearch as bs
from beamsearch import unroll
from scipy.misc import imread
#import spacy
#import wmd
from keras.models import Model
from nltk.translate.bleu_score import sentence_bleu

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

def process():
     
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
    return d


def generate_k_best(model, enc_map, dec_map, img, k=4, max_len=10):
    #img = img_map[]
    ans = bs(model, enc_map, dec_map, img, k, max_len)
    gen = []
    for x in ans[1:-1]:
        if x == 1 : break
        if x != 0 : gen.append(dec_map[x])
    return  ' '.join(gen)

def eval_human(model, img_map, df_cap, enc_map, dec_map, k=4, size=1, max_len=10):
    for idx in np.random.randint(df_cap.shape[0], size=size):
        row = df_cap.iloc[idx]
        cap = eval(row['captions'])
        img_id = row['image_id']
        img = img_map[img_id]
        gen = generate_k_best(model, enc_map, dec_map, img, k=k, max_len=max_len)
        print('[{}]'.format(img_id))
        print('[generated] {}'.format(gen))
        print('[groundtruth] {}'.format(' '.join([dec_map[cap[i]] for i in range(1,len(cap)-1)])))

def generateResults(test_img,model):
    results = []
    hyp1 = open("results/hyp.txt", "a")

    results_beam = []
    images = '/home/manish/Image-Captioning-master/Flicker8k_Dataset/'
    dec_map = pickle.load(open("dec_map.pkl", 'rb'))
    enc_map = pickle.load(open("enc_map.pkl", 'rb'))
    img_test = pickle.load(open("../encoded_images_test_inceptionV3.p", 'rb'))
    for i in test_img:
        #results.append(predict_captions(i,model))
        #results_beam.append(beam_search_predictions(7,i,model))
        img = img_test[i[len(images):]]
        results_beam.append(generate_k_best(model, enc_map, dec_map, img, 3,53))
        print(i,results_beam[-1])
        hyp1.write(results_beam[-1]+"\n")

    return results_beam

# In[87]:
def getLabels(test_img,test_dict):
   labels = []
   ref1 = open("results/ref1.txt", "a")
   ref2 = open("results/ref2.txt", "a")
   ref3 = open("results/ref3.txt", "a")
   ref4 = open("results/ref4.txt", "a")
   ref5 = open("results/ref5.txt", "a")
   for i in test_img:
       temp = []
       ref = test_dict[i]
       for j in test_dict[i]:
           temp.append(j.split())
       ref1.write(ref[0]+"\n")
       ref2.write(ref[1]+"\n")
       ref3.write(ref[2]+"\n")
       ref4.write(ref[3]+"\n")
       ref5.write(ref[4]+"\n")
       labels.append(temp)
       
   ref1.close()
   ref2.close()
   ref3.close()
   ref4.close()
   ref5.close()
   return labels
'''
def getLabels(test_img,test_dict):
    labels = []
    for i in test_img:
        temp = []
        for j in test_dict[i]:
            temp.append(j.split())
    
        labels.append(temp)
    return labels
'''
from nltk.translate.bleu_score import corpus_bleu

def bleu_1__average_score(test_img, labels,splitResults):
    score = []
    score1 = []
    for r in range(test_img):
        reference = labels[r]
        candidate = splitResults[r]
        score.append(sentence_bleu(reference, candidate))
        score1.append(sentence_bleu(reference, candidate,weights = (1,0,0,0)))
    print("BLEU-1 average score:", np.mean(score))
    print("BLEU-1 average score:", np.mean(score1))


def bleu_corpus_score(labels,split_results):
    references = labels
    candidates = split_results
    score = corpus_bleu(references, candidates)
    print("BLEU corpus score:", score)
    score = corpus_bleu(references, candidates,weights = (0,1,0,0))
    print("BLEU corpus score:", score)
    score = corpus_bleu(references, candidates,weights = (0,0,1,0))
    print("BLEU corpus score:", score)
    score = corpus_bleu(references, candidates,weights = (1,0,0,0))
    print("BLEU corpus score:", score)
    score = corpus_bleu(references, candidates,weights = (0,0,0,1))
    print("BLEU corpus score:", score)   
     
def split_data(l,img,lengthImages):
    temp = []
    for i in img:
        if i[lengthImages:] in l:
            temp.append(i)
    return temp

def processTestData(filename,d):

    test_images = set(open(filename, 'r').read().strip().split('\n'))
    images = '../Flicker8k_Dataset/'
    
    img = glob.glob(images+'*.jpg')
    test_img = split_data(test_images,img,len(images))
    test_dict = {}
    for i in test_img:
        if i[len(images):] in d:
            test_dict[i] = d[i[len(images):]]
    print (len(test_img))
    return test_img,test_dict


if __name__ == '__main__':
    #path = sys.argv[1]
    dec_map = pickle.load(open("dec_map.pkl", 'rb'))
    enc_map = pickle.load(open("enc_map.pkl", 'rb'))

    dict = process()
    
    #img_train = pickle.load(open("/general/home/ronakchaudhary132199/Image-Captioning-master/encoded_images_inceptionV3.p", 'rb'))
    img_test = pickle.load(open("../encoded_images_test_inceptionV3.p", 'rb'))
   
    
    
    model = image_caption_model(vocab_size = 8256,clipnorm=1.)
    model.load_weights("weights/v1.0.0_6_39_1524863089.8904815.h5")
    #model.load_weights("")")

    #eval_human(model, img_train, df_cap, enc_map, dec_map, k=1, size=40, max_len=13)
    '''
    print("____________________________________________________________________________________________________________________________")
    model.load_weights("/general/home/manish.singhal/attention/weights/v1.0.0_60_69_1523845852.7416637.h5")
    print(generate_k_best(model, enc_map, dec_map, img1, 3,13))
    print(generate_k_best(model, enc_map, dec_map, img2, 3,13))
    print(generate_k_best(model, enc_map, dec_map, img3, 3,13))
    print(generate_k_best(model, enc_map, dec_map, img4, 3,13))
    print(generate_k_best(model, enc_map, dec_map, img5, 3,13))
    '''
    
##===============================================================================================###

    test_images_file = '../processed_files/Flickr_8k.testImages.txt'
    test_img,test_dict = processTestData(test_images_file,dict)

    results_beam = generateResults(test_img,model)
    labels = getLabels(test_img,test_dict)
  
    split_results_beam = []
    for i in results_beam:
        split_results_beam.append(i.split())
    #bleu_1__average_score(len(test_img), labels,split_results_beam)
    #bleu_corpus_score(labels,split_results_beam)

