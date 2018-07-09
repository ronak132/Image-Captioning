#import requirements
import os
import sys
scriptpath = "/home/manish.singhal/Image-Captioning-master/caption.py"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))


import caption
import glob
#from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.preprocessing import image
import nltk
import random
from scipy.misc import imread
#import spacy
#import wmd
from keras.models import Model
from nltk.translate.bleu_score import sentence_bleu

def predict_captions(image,model):
    start_word = ["<start>"]
    
    #encoding_test = pickle.load(open('/home/ronakchaudhary132199/Image-Captioning-master/encoded_images_test_inceptionV3.p', 'rb'))
    #print ('\n=============Here=======\n')

    while True:
        par_caps = [cg.word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=cg.max_cap_len, padding='post')
        e = cg.encode(image)
        preds = model.predict([np.array([e]), np.array(par_caps)])
        word_pred = cg.idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > cg.max_cap_len:
            break
            
    return ' '.join(start_word[1:-1])






def plot_image(image):
    plt.imshow(imread(image))
    plt.show()
    plt.clf()

def generate_captions(samples,model,test_img):
    for idx in samples:
        img = test_img[idx]
        #print ("here ia m ")
        #plot_image(img)
        print("Generated Caption:", predict_captions(img,model))


#def testing(model,test_img):
# ## Good Results
    #good_results=[340,257,642,996,787,35,846,570,277,482]
    #generate_captions(good_results,model,test_img)

# ## Bad Results
    #bad_results=[180,432,209,102,590,839,571,746,452,312]
    #generate_captions(bad_results,model,test_img)
	#r = random.randint(0, len(test_img) - 1)
    #plot_image(test_img[r])
    #print (labels[r])
    #print (split_results[r])


# ## Testing

# In[23]:

def generateResults(test_img,model):
    results = []
    results_beam = []

    for i in test_img:
        #results.append(predict_captions(i,model))
        results_beam.append(beam_search_predictions(7,i,model))
        print(i,results_beam[-1])
    return results,results_beam

# In[87]:

def getLabels(test_img,test_dict):
    labels = []
    for i in test_img:
        temp = []
        for j in test_dict[i]:
            temp.append(j.split())
    
        labels.append(temp)
    return labels
	
from nltk.translate.bleu_score import corpus_bleu

def bleu_corpus_score(labels,split_results):
    references = labels
    candidates = split_results
    score = corpus_bleu(references, candidates)
    print("BLEU corpus score:", score)


def bleu_1__average_score(test_img, labels,splitResults):
    score = []
    for r in range(len(test_img)):
        reference = labels[r]
        candidate = splitResults[r]
        score.append(sentence_bleu(reference, candidate,weights = (1,0,0,0)))
    print("BLEU-1 average score:", np.mean(score))

def bleu_2__average_score(test_img, labels,splitResults):
    score = []
    for r in range(len(test_img)):
        reference = labels[r]
        candidate = splitResults[r]
        score.append(sentence_bleu(reference, candidate,weights = (0,1,0,0)))
    print("BLEU-2 average score:", np.mean(score))
	
def bleu_3__average_score(test_img, labels,splitResults):
    score = []
    for r in range(len(test_img)):
        reference = labels[r]
        candidate = splitResults[r]
        score.append(sentence_bleu(reference, candidate,weights = (0,0,1,0)))
    print("BLEU-3 average score:", np.mean(score))
	
def bleu_4__average_score(test_img, labels,splitResults):
    score = []
    for r in range(len(test_img)):
        reference = labels[r]
        candidate = splitResults[r]
        score.append(sentence_bleu(reference, candidate,weights = (0,0,0,1)))
    print("BLEU-4 average score:", np.mean(score))
# In[98]:


def beam_search_predictions( beam_index,image, final_model):
    start = [cg.word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < cg.max_cap_len:
        temp = []
        for s in start_word:
            #par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            #e = encoding_test[image[len(images):]]
            #preds = final_model.predict([np.array([e]), np.array(par_caps)])
            #par_caps = [cg.word2idx[i] for i in start_word]
            par_caps = sequence.pad_sequences([s[0]], maxlen=cg.max_cap_len, padding='post')
            e = cg.encode(image)
            preds = final_model.predict([np.array([e]), np.array(par_caps)])            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [cg.idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption



def compute_WMD_score(test_img,test_dict,results):
    nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)

    scores = []
    for i, j in enumerate(test_img):
        res = nlp(results[i])
        for k in test_dict[j]:
            temp = nlp(k)
            scores.append(temp.similarity(res))

    print ("WMD:", np.mean(scores))

def processTestData(filename,d):

    test_images = set(open(filename, 'r').read().strip().split('\n'))
    images = '../Flicker8k_Dataset/'
    
    img = glob.glob(images+'*.jpg')
    test_img = cg.split_data(test_images,img,len(images))
    test_dict = {}
    for i in test_img:
        if i[len(images):] in d:
            test_dict[i] = d[i[len(images):]]
    print (len(test_img))
    return test_img,test_dict

cg = caption.Caption()

def main():
    print ('Manish') 
    #cg = Caption() 
    model = cg.create_model()
    #model.load_weights('/home/manish.singhal/time_inceptionV3_manish.h5')
    model.load_weights("weights/weights-improvement_epoch50_adam-70.hdf5")

    test_images_file = '../processed_files/Flickr_8k.testImages.txt'
    test_img,test_dict = processTestData(test_images_file,cg.d)
	#Do Testing
    #testing(model,test_img)
    results,results_beam = generateResults(test_img,model)
    labels = getLabels(test_img,test_dict)
    split_results = []
    for i in results:
        split_results.append(i.split())
    split_results_beam = []
    for i in results_beam:
        split_results_beam.append(i.split())
    #bleu_1__average_score(test_img, labels,split_results)
    #bleu_2__average_score(test_img, labels,split_results)
    #bleu_3__average_score(test_img, labels,split_results)
    #bleu_4__average_score(test_img, labels,split_results)
    print ('\n\n==========Beam Scores=============\n\n')
    bleu_1__average_score(test_img, labels,split_results_beam)
    bleu_2__average_score(test_img, labels,split_results_beam)
    bleu_3__average_score(test_img, labels,split_results_beam)
    bleu_4__average_score(test_img, labels,split_results_beam)



if __name__ == '__main__':
    main()
