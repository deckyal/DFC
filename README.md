# DFC
This the repository of our Heatmap-Guided Deep Kinship Family Classification in The Wild

<b>Requirements : </b>
1. PyTorch GPU 
2. Tensorflow GPU
3. RFIW Dataset :
  a. https://web.northeastern.edu/smilelab/fiw/download.html 
  b. training.list and val.list from https://competitions.codalab.org/competitions/20196#participate-get_data (you may need to register)

<b>Usage : </b>
1. Put the FIW data on 
2. Put the training.list and val.list on 

<b>Replicate the 2nd challenge of RFIW (https://web.northeastern.edu/smilelab/RFIW2019/) 
  
python DFC.py --test=1
  
<b> Train on your own dataset </b>

python DFC.py --test=0
