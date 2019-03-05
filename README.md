# DFC
This the repository of our Heatmap-Guided Balanced Deep Convolution Networks for Family Classification in the Wild.

<b>Requirements : </b>
1. PyTorch GPU  https://pytorch.org/
2. Tensorflow GPU https://www.tensorflow.org/install
3. FIW Dataset 
    a. : from https://competitions.codalab.org/competitions/20196#participate-get_data (you may need to register)
    b. original data from https://web.northeastern.edu/smilelab/fiw/download.html
  
This repository holds : 
1. The image normalizer from : https://github.com/deckyal/FADeNN
2. The facial landmark localiser from : https://github.com/deckyal/RT

<b>Usage : </b>

<b>Replicate the 2nd challenge of RFIW (https://web.northeastern.edu/smilelab/RFIW2019/) </b>

<b>Preparations</b>

1. Put the FIW data on images/ (ex : /DFC/images/FIDs/F0001/MID1)
2. Put the test_no_labels.list on the main folder :DFC/ 
3. Run the LandmarkingHeatmap to get the corresponding denoised image and the facial heatmaps
  
python Reproduce.py 

The corresponding CSV will be on the ./models

<b>Replicate the 5 cross validation test of Family classifiaiton on th wild FIW https://web.northeastern.edu/smilelab/fiw/benchmarks.html </b>
  
<b>Preparations</b>

1. Put the FIWNews data on images/ (ex : /DFC/images/FIDsNew/F0001/MID1)
2. Download the five validation split from : https://web.northeastern.edu/smilelab/fiw/download.html and put on the /DFC/cl-info folder
2. Run the LandmarkingHeatmap to get the corresponding denoised image and the facial heatmaps
  
python ReproduceFIW.py -testFold=0

-testFold signifies the fold test (0~4). The corresponding CSV will be on the ./models

Citation : 
D. Aspandi, O. Pujol, and X. Binefa. Heatmap-Guided Balanced Deep Convolution Networks for Family Classification in the Wild. In Recognizing Families In the Wild (RFIW) workshop in conjunction with FG 2019, In Press
