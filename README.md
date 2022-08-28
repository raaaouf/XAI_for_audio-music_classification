# XAI_for_Music_genre_classification

This repository contains the source code and implementation material for the project ***"Implementation of the Hybride Explainable artificial intelligence (XAI) method for music recognition"***.

The goal of our work is to implement an algorithm to explain the predictions of music recognition systems. The repository contains the code. Two versions of explanations (temporal and time-frequency) can be generated using the repository code. Details about the constituents of each XAI model and how to generate explanations is provided in the "Readme" file located at each sub-directory of the official implementations.

This work is divided into three parts, each containing the results of the methodological phases discussed in the RTD. 


Note: some code snippets were adopted or are based on other repositories/sources, which is stated in the concerned cells. 


## Requirements
* All experiments were conducted with laboratory server (recommended high Ram and GPU)
* part 3 requires the outputs of part 1 and 2
* python: numpy, pandas, tensorflow, keras, sklearn, IPython, lime, skimage, sys, math, time, logging, glob, os, scipy, h5py, json, librosa, multiprocessing, urllib, io, zipfile, pickle, matplotlib, PIL

## Dataset
The dataset used is [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (the famous GTZAN dataset, the MNIST of sounds)

The GTZAN dataset contains 1000 audio files. 
Contains a total of 10 genres, each genre contains 100 audio files

1.Blues 

2.Classical

3.Country

4.Disco   

5.Hip-hop 

6.Jazz   

7.Metal 

8.Pop 

9.Reggae 

10.Rock

## PART 1 BLACK BOX models training for MGR
### GOAL: 
In this PART are presented the use case we shoosed for audio classification followed
by a precise description of the audio representations, EDA and audio prossesing using LIBROSA library for creating the data samples used as input in our proposed XAI technique and finally we present the neural network we implimented from scratch as a baseline, training phase of the
network  and the pretrained VGG network using Tranfer learning, used as black-box model to produce the explanations

### Usage
1. Clone or download this repository into your  root folder (if you use a subfolder, you have to adjust the root_path in the ipynb notebooks)
2. Run all cells of the first script in order to:
- Download dataset
- Data processing
- EDA
- Create spectrograms (.hdf5 files)
- Create splits
- Train models
- Evaluate models (accuracy, confusion matrix and loss plots)
- Transfer learning with VGG16

### Results
Experiment results showed that validation accuracies and losses are similaire in both baseline and vgg model

| Model | BASELINE    | VGG16    |
| :---:   | :---: | :---: |
| Accuracy | 0,95	| 0,93   |
| LOSS | 0,25   | 0,24   |



## PART 2 SHAP LIME for LOCAL XAI
### LIME:
Lime is able to explain any black box classifier, with two or more classes. All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a probability for each class. Support for scikit-learn classifiers is built-in.

### SHAP:
SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions

### Usage
- run script" of the second part
- pip install shap
- pip install LIME

### Results
- You can use the notebook *Local_XAI_for_MGR_.ipynb* to plot and analyse the results.
- Statistical results showed that SHAP MFCC features contribution correlations are more related to the modele predictions
- We compared the XAI results and modele precision using mean contributions precision metric and in differnt iterations we found:
  - Classifier: à.95
  - SHAP: 0.86
  - LIME: 0.70
- The experimental results do not show stronger attributions to classes using LIME this is because of its instable nature in nature (see report for more details).
- Visual results can be prouved by choosing other parameters that are much relevent to the task ( see future work section in report).






## PART 3 ALE for globale XAI
Accumulated local effects(ALE) aims to highlight the effects that a specific features have on the predictions of a machine learning model by partially isolating the effects of other features. The resulting ALE explanation is centered around the mean effect of the feature, such that the main feature effect is compared relative to the average prediction of the data.

### Usage
- pip install ALE
- converting the data from 2D to a 1D array by averaging the rows
### Results
The experiments results show the Accumulated Local Effects (ALE) plots for the audio classification model trained on the music dataset for the respective features shosen by SHAP.


So, in the plots (see ALE section in report) we can see how ALE shows the importance of feature at the class level by showing in which interval the feature is important for the respective class.
In this way, Accumulated Local Effects (ALE) helps in looking at the effects that specific features have on the predictions of a machine learning model.
