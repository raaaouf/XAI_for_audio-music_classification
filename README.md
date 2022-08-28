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




## PART 2 SHAP LIME for LOCAL XAI
### LIME:

### SHAP:
SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions

### Usage
- run script" of the second part
- pip install shap
- pip install LIME

### Results
You can use the notebook *Local_XAI_for_MGR_.ipynb* to plot and analyse the results.






## PART 3 ALE for globale XAI
### Usage

### Results

*it is highly recommended to use Google Colab Pro+ with GPU runtime and high RAM setup, more information at the beginning of the script!



## Folder structure (will be created with the scripts):

    XAI_spec_AudioMNIST				# root folder of the repository
    ├──AudioMNIST-master				# downloaded dataset
    |	├──data
    |	|	├──01				# contains .wav files (raw data) of participant 01
    |	|	├──...				# same for each of the participants
    |	|	└──audioMNIST_meta.txt		# contains meta information on the participants							
    |	└──...					# other files and folders are irrelevant
    ├──results
    |	├──evaluation
    |	|	├──confusionMatrix_digit_0.csv	# contains the confusion matrix for label digit and fold 0
    |	|	├──...				# same for each label and fold, additionally mean for each label
    |	|	├──evalutation_digit_0.csv	# contains performance indicators (accuracy) for label digitand fold 0	
    |	|	└──...				# same for each label and fold, additionally mean for each label
    |	├──history
    |	|	├──AlexNet_digit_0.pkl		# contains the history of label digit and fold 0
    |	|	└──...				# same for each label and fold
    |	├──models
    |	|	├──AlexNet_digit_0.h5		# contains the model of label digit and fold 0
    |	|	└──...				# same for each label and fold
    |	├──plots
    |	|	├──loss				# contains the loss plots as .png files
    |	|	├──spectrograms			# contains the spectrograms as .png files
    |	|	├──waveform			# contains the waveform plots as .png files
    |	|	└──xai				# contains the explanations as .png files for Grad-CAM and LIME
    |	└──predictions
    |		├──predictions_digit_0.csv	# contains the predictions of label digit and fold 0
    |		└──...				# same for each label and fold	
    ├──spectrograms
    |	├──01					# contains .hdf5 files (spectrograms) of participant 01
    |	└──...					# same for each of the participants
    ├──splits					# here: split = fold
    |	├──AlexNet_digit_0_test.txt		# contains paths to spectrograms (.hdf5 files) for label digit, fold 0 and testsplit
    |	└──...					# same for each label, fold and split
    ├──01_Preprocessing_Training_Evaluation.ipynb	# script 1 contains code for downloading dataset, creating spectrograms (.hdf5 files), training models, 
    ├──02_XAI_methods.ipynb				# script 2 contains code for creating outputs as .png files (waveform plots, spectrograms, Grad-CAM, LIME) 
    ├──events.log					# contains logs, produced while running the code
    ├──LICENSE
    └──README.md

