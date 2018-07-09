import os, sys
import pickle 
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np

from keras.models import load_model
from utils import *
from model import image_caption_model
from joblib import Parallel, delayed
import time
from keras.utils.layer_utils import print_summary
#import matplotlib.pyplot as plt

import os
import sys
scriptpath = "model.py"
sys.path.append(os.path.abspath(scriptpath))


def gen_batch_in_thread(img_map, df_cap, vocab_size, n_jobs=4,
        size_per_thread=32):
    imgs , curs, nxts, seqs, vhists = [], [], [], [], []
    returns = Parallel(n_jobs=4, backend='threading')(
                            delayed(generate_batch)
			    (img_map, df_cap, vocab_size, size=size_per_thread) for i in range(0, n_jobs))
    #return = generate_batch(img_train, df_cap, vocab_size, size=size_per_thread)

    for triple in returns:
        imgs.extend(triple[0])
        curs.extend(triple[1])
        nxts.extend(triple[2])
        seqs.extend(triple[3])
        vhists.extend(triple[4])

    return np.array(imgs), np.array(curs).reshape((-1,1)), np.array(nxts), \
            np.array(seqs), np.array(vhists)

def generate_batch(img_map, df_cap, vocab_size, size=32, max_caplen=53):
    imgs, curs, nxts, seqs, vhists = [], [], [], [], []
    word2idx = pickle.load(open("enc_map.pkl", 'rb'))

    
    c = [i for i in df_cap['captions']]
    #print(c[-2])
    
    img_ = [j for j in df_cap['image_id']]
    #print(img_[-1])

    iter = df_cap.iterrows()
    #for p in range(32):
    for idx in np.random.randint(df_cap.shape[0], size=size):
        
        #x = next(iter)
        cap = c[idx].split()
        
        image= img_[idx]

        img = img_map[image]

        vhist = np.zeros((len(cap)-1, vocab_size))

        for i in range(1, len(cap)):
            #print(cap[i])
            ii = word2idx[cap[i]]
            ii_1 = word2idx[cap[i-1]]
            seq = np.zeros((max_caplen))
            nxt = np.zeros((vocab_size))
            nxt[ii] = 1
            curs.append(ii_1)
            seq[i-1] = 1

            if i < len(cap)-1:
                vhist[i, :] = np.logical_or(vhist[i, :], vhist[i-1, :])
                vhist[i, ii_1] = 1

            nxts.append(nxt)
            imgs.append(img)
            seqs.append(seq)

        vhists.extend(vhist)

    return imgs, curs, nxts, seqs, vhists

if __name__ == '__main__':


    dec_map = pickle.load(open("dec_map.pkl", 'rb'))
    enc_map = pickle.load(open("enc_map.pkl", 'rb'))
#"/general/home/manish.singhal/Image-Captioning-master/encoded_images_inceptionV3.p"
    img_train = pickle.load(open("../encoded_images_inceptionV3.p", 'rb'))
    
    img_test = pickle.load(open("../encoded_images_test_inceptionV3.p", 'rb'))
    print (len(img_train))
    df_cap = pd.read_csv("../processed_files/flickr8k_training_dataset.txt", delimiter = '\t')

    vocab_size = len(dec_map)
    print("vocab sixe is",vocab_size)
    embedding_matrix = generate_embedding_matrix('pre_trained/glove.6B.100d.txt', dec_map)
    model = image_caption_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)

    if len(sys.argv) >= 2:
        print('load weights from : {}'.format(sys.argv[1]))
        model.load_weights(sys.argv[1])

    # insert ur version name here
    version = 'v1.0.0'
    batch_num = 40
    #print_summary(model.layers)
    model.summary()
    hist_loss = []
   # history = pickle.load(open("/general/home/manish.singhal/attention/weights/history.pkl", 'rb'))
    for i in range(0,10 ):
        for j in range(1, batch_num+1):
            s = time.time()
	    # 64 x 128 = 8192 images per batch.
	    # 8 x 32 = 256 images for validation.
            img1, cur1, nxt1, seq1, vhists1 = gen_batch_in_thread(img_train, df_cap,
                                    vocab_size, n_jobs=16, size_per_thread=128)
            img2, cur2, nxt2, seq2, vhists2 = gen_batch_in_thread(img_train, df_cap, 
                                    vocab_size, n_jobs=1, size_per_thread=32)
                              
            history = model.fit([img1, cur1, seq1, vhists1], nxt1, batch_size=1024, nb_epoch=1, verbose=1,
                                    validation_data=([img2, cur2, seq2, vhists2], nxt2), shuffle=True)

            print("epoch {0}, batch {1} - training loss : {2}, validation loss: {3}"
                    .format(i, j, history.history['loss'][-1], history.history['val_loss'][-1]))
	    # record the training history
            mdl_path = "weights/"
            #print('check point')
            #m_name = "{0}{1}_{2}_{3}_{4}.h5".format(mdl_path, version, i, j, time.time())
            #model.save_weights(m_name)
            hist_loss.extend(history.history['loss'])
            

            if j % int(batch_num-1) == 0 and  (i %3 ) == 0:
                print('check point')
                m_name = "{0}{1}_{2}_{3}_{4}.h5".format(mdl_path, version, i, j, time.time())
                model.save_weights(m_name)
                pickle.dump({'loss':hist_loss}, open(mdl_path+ 'history.pkl', 'wb'))
