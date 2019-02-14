# DFC
This the repository of our Coming Soon

<b>Requirements : </b>
1. PyTorch GPU 
2. Tensorflow GPU
3. RFIW Dataset :
  a. https://web.northeastern.edu/smilelab/fiw/download.html 
  b. training.list and val.list from https://competitions.codalab.org/competitions/20196#participate-get_data (you may need to register)
  
This repository holds : 
1. The denoising from : https://github.com/deckyal/FADeNN
2. The facial landmark localiser from : https://github.com/deckyal/RT

<b>Usage : </b>
1. Put the FIW data on 
2. Put the training.list and val.list on 

<b>Replicate the 2nd challenge of RFIW (https://web.northeastern.edu/smilelab/RFIW2019/) </b>
  
python DFC.py --test=1
  
<b> Train on your own dataset </b>

python DFC.py --test=0

Citation : 
Coming soon
