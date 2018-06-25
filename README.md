# CSCE-633 Machine Learning Project
Implemented a deep learning model that automatically generates image captions.

Environment Dependencies for train and test:
=========================
- Python 3.6 <br>
- Keras-gpu <br>
- Tensorflow-gpu <br>

Library Dependencies for train and test:
=====================
- nltk <br>
- matplotlib <br>
- numpy <br>
- pickle <br>
- sys <br>
- tqdm <br>
- pandas <br>
- glob <br>
- pillow <br>
- h5py <br>

Library and Environment Dependencies for caluclating the scores:
=========================
- java 1.8.0 <br>
-  python 2.7 <br>
- click 6.3 <br>
- nltk 3.1 <br>
- numpy 1.11.0 <br>
- scikit-learn 0.17 <br>
- gensim 0.12.4 <br>
- Theano 0.8.1 <br>
- scipy 0.17.0 <br>

## Dataset
The model has been trained and tested on Flickr8k dataset[2]. There are many other datasets available that can used as well like:	
- Flickr30k
- MS COCO
- SBU
- Pascal

## Usage

After the requirements have been installed, the process from training to testing is fairly easy. The commands to run:
1) For CNN+LSTM model (without attention), go to folder CNN+LSTM and then:

    a. `python train_model.py`<br>
    b. `python test.py`<br>
1) For CNN+LSTM model (without attention), go to folder CNN+LSTM and then:

    a. `python train.py`<br>
    b. `python evaluate.py`<br>
    
Directory Tree: (output of 'tree -L 3')
-------------------
```bash
.
├── attention_model
│   ├── beamsearch.py
│   ├── dec_map.pkl
│   ├── enc_map.pkl
│   ├── evaluate.py
│   ├── model.py
│   ├── pre_trained
│   │   └── glove.6B.100d.txt
│   ├── __pycache__
│   │   ├── beamsearch.cpython-35.pyc
│   │   ├── model.cpython-35.pyc
│   │   └── utils.cpython-35.pyc
│   ├── results
│   ├── train.py
│   ├── utils.py
│   └── weights
│       └── v1.0.0_6_39_1524863089.8904815.h5
├── CNN+LSTM
│   ├── caption.py
│   ├── caption.pyc
│   ├── __pycache__
│   │   └── caption.cpython-35.pyc
│   ├── test.py
│   ├── train_model.py
│   ├── unique.p
│   ├── weights
│   │   └── weights-improvement_epoch50_adam-70.hdf5
│   └── weights-improvement_epoch50_adam-70.hdf5
├── encoded_images_inceptionV3.p
├── encoded_images_test_inceptionV3.p
├── Flicker8k_Dataset
└── processed_files
    ├── Flickr_8k.devImages.txt
    ├── Flickr8k.lemma.token.txt
    ├── Flickr_8k.testImages.txt
    ├── Flickr8k.token.txt
    ├── Flickr_8k.trainImages.txt
    ├── flickr8k_training_dataset.txt
    └── unique.p
   ```
Git references:
---------------
- https://github.com/Maluuba/nlg-eval
- https://github.com/yashk2810/Image-Captioning
## References 
[1] Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)

[2]	Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. Collecting Image Annotations Using Amazon's Mechanical Turk. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon's Mechanical Turk.
