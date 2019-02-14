'''
Created on Oct 5, 2018

@author: deckyal
'''

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch.nn.functional as F

import file_walker
from utils import *
from config import *

from NetworkModels import LogisticRegression,LogisticRegressionC,LogisticRegressionShort,VAERegression
from FacialDataset import FamilyDataset
from CenterLoss import CenterLoss

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 : 
        nn.init.normal_(m.weight.data, 0.0,0.02)
    elif classname.find('BatchNorm') != -1 :  
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.normal_(m.bias.data,0)
    elif classname.find('Linear') != -1 : 
        nn.init.normal_(m.weight.data,1.0,0.001)
        m.bias.data.fill_(0.001)
        print('initting')
    #print(m)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting : 
        for param in model.parameters() : 
            param.requires_grad = False


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# validatePre(modelType=1,useBCE =True,fr =0, wh = 0, wvl = 0, dirName = 'model-12DecS/')

parser = argparse.ArgumentParser()
parser.add_argument('-type', nargs='?', const=1, type=int, default=1)#0,1,2
parser.add_argument('-skip', nargs='?', const=1, type=int, default=0)#0,1
parser.add_argument('-load', nargs='?', const=1, type=int, default=1)#0,1
parser.add_argument('-useC', nargs='?', const=1, type=int, default=0)#0,1

parser.add_argument('-fr', nargs='?', const=1, type=int, default=0)#0,1
parser.add_argument('-bce', nargs='?', const=1, type=int, default=1)#0,1

parser.add_argument('-wh', nargs='?', const=0, type=int, default=1)#0,1
parser.add_argument('-wvl', nargs='?', const=0, type=int, default=0)#0,1
parser.add_argument('-wcl', nargs='?', const=0, type=int, default=0)#0,1
parser.add_argument('-nmlz', nargs='?', const=0, type=int, default=0)#0,1

parser.add_argument('-rd', nargs='?', const=0, type=int, default=1)#0,1
parser.add_argument('-iv', nargs='?', const=0, type=int, default=1)#0,1
parser.add_argument('-ud', nargs='?', const=0, type=int, default=1)#0,1
parser.add_argument('-typeTrain', nargs='?', const=2, type=int, default=2)#0,1,2#0 is normal, 1 is train image Re, 2 is train image Se, 3 is incres
    


args = parser.parse_args()
args = parser.parse_args()

type = args.type
skip = args.skip 
loadModel = args.load
useCosDist = args.useC
fr = args.fr
rd = args.rd
useBCE = args.bce
wh = args.wh
wvl = args.wvl
iv = args.iv
ud = args.ud
wcl = args.wcl
nmlz = args.nmlz


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ])

debug = False

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ]),
    'NormalizeM': [.5,.5]
}


def validatePre(modelType=1,useBCE=True, fr=1, wh = 0, wvl = 0,wcl = 0, nmlz = 0,includeVal = True,dirName='models/',
    use_best = False):
    
    add_name = '-reduce'
    
    if add_name is not None : 
        rl = True
    else : 
        rl = False
    
    
    if fr : 
        batch_size = 120#150#300
    else : 
        batch_size = 300#300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    listCSVVal = ['test_no_labels.list']
    isTest = True
    
    useCleanedDataset = True;
     
    if useCleanedDataset :
        dir_add = "-GCAE"
    else : 
        dir_add = ""
        
    dirImage = curDir+ 'images'+dir_add+'/FIDs/'
    dirList =  ['landmarks'+dir_add,'heatmaps'+dir_add]
    
        
    IDV = FamilyDataset(data_list = listCSVVal,transforms=data_transforms,dirImage = dirImage,withLandmark=True,withHeatmap = wh,useIT = True,multiLabel = True,convertToOneHot = False,dirList = dirList,isTest = isTest,rl = rl)
    dataloaderVal = torch.utils.data.DataLoader(dataset = IDV, batch_size = batch_size, shuffle = False)
        
    load = True
    loadClassifier = True
    
    #Create the Generator 
    import csv
    isVae = False
    input_size = 512
    
    num_classes = 1
    if useBCE : 
        if rl :  
            num_classes = 540
        else : 
            num_classes = 1000
    
    if fr == 1 : 
        freezeExtractor  = True
    else : 
        freezeExtractor = False 
    
    #modelType = 0 #0 resnet, 1 senet
    
    N_IDENTITY = 8631
    num_inter =input_size
    
    import resnet as ResNet
    import senet as SENet
    
    if modelType == 0:
        net = ResNet.resnet50(num_classes=N_IDENTITY)
    else:
        net = SENet.senet50(num_classes=N_IDENTITY)


    if not load :  
        if modelType == 0 :
            net.load_state_dict(torch.load(curDir+'src/'+'resnet50.pt'))
            
        else : 
            net.load_state_dict(torch.load(curDir+'src/'+'senet50.pt'))
        
        print(net)
        
        if freezeExtractor : 
            set_parameter_requires_grad(net, True)
        
        if wh : 
            net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)#4 using heatmap
            
        num_ftrs = net.fc.in_features 
        net.fc = nn.Linear(num_ftrs, num_inter)

    if modelType == 0 : 
        fileName = "m-pf-R"
        fileRName = 'm-pf-R-C'
    else : 
        fileName = "m-pf-S-C"
        fileRName = 'm-pfc-S-C'

    if freezeExtractor : 
        fileName+='-F'
        fileRName += '-F'
        
    if not useBCE : 
        fileName +='-nBCE'
        fileRName +='-nBCE'
    
    if wh : 
        fileName +='-wh'
        fileRName +='-wh'
    
    if wvl : 
        fileName +='-wl'
        fileRName +='-wl'
        
    if rd : 
        fileName +='-rd'
        fileRName +='-rd'
        
    if not includeVal : 
        fileName +='-normal'
        fileRName +='-normal'
    
    
    fileName +=add_name
    fileRName +=add_name
    
    if wcl : 
        fileName +="-wcl"
        fileRName +="-wcl"

    if nmlz: 
        fileName +="-nmlzed"
        fileRName +="-nmlzed"
        
    if use_best : 
        fileName +='-b'
        fileRName +='-b'
        
    err_file = curDir+fileName+".txt"
    
    f = open(err_file,'w+')
    f.write("err : ")
    f.close()

    if isVae : 
        net = DVAE()
    if load : 
        print('loading cae',curDir+'src/'+dirName+fileName+'.pt')
        if wh : 
            net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)#4 using heatmap
            
        num_ftrs = net.fc.in_features
        
        if wvl : 
            net.fc = VAERegression(num_ftrs, num_inter)
        else : 
            net.fc = nn.Linear(num_ftrs, num_inter)
            
        net.load_state_dict(torch.load(curDir+dirName+fileName+'.pt'))
    net = net.cuda()
    
    #The classifier
    if type >1 : 
        model_lg = LogisticRegressionC(input_size,136*2, num_classes)
    else : 
        if rd : 
            model_lg = LogisticRegressionShort(input_size, num_classes)
        else :
            model_lg = LogisticRegression(input_size, num_classes)
        
    if loadClassifier : 
        print("loading cl")
        model_lg.load_state_dict(torch.load(curDir+dirName+fileRName+'.pt'))
    else : 
        model_lg.apply(weight_init)
        
    model_lg.cuda()
    
    print('Starting Validation Loop : ') 
    
    #do validation 
    net.eval()
    model_lg.eval()

    gacc = 0; 
    glen = 0; 
    
    ofile  = open(curDir+dirName+fileName+'-'+fileRName+'-val.csv', "w")
    spamwriter = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    #spamwriter.writerow(['p1', 'p2', 'Label'])
    
    correct = 0
    total = 0
    acc = 0
    wrong = 0
    
    for i, (x1,gt,htmp,lm,dta1) in enumerate(dataloaderVal) :
        
        with torch.no_grad(): 
            
            if wh :
                #print(x1.shape,htmp)
                inputx1 = torch.cat((x1,htmp[0]),1).cuda()
            else :
                x1 = x1.cuda()
                inputx1 = x1
                
            gt = gt.cuda().squeeze()
            
            net.zero_grad()
            model_lg.zero_grad()
            
            immediate = net(inputx1)
            mu,log_var = None, None
            
            if wvl : 
                xe = immediate[0]
                mu,log_var = immediate[1],immediate[2]
            else : 
                xe = immediate 
            #print(xe.shape)
            
            labels = model_lg(xe,useBCE = False)
            
            if useBCE :
                print('using BCE')
                _,out = torch.max(labels,1)
            else : 
                print("not using BCE")
                out = labels
            
            '''t = torch.Tensor([0.5]).cuda()  # threshold
            out = (labels > t).float() * 1'''
            
            print('out : ',out,'gt : ',gt,"labels ",labels)
            
            #preprocess label. 
            labelMapping = np.load(curDir+'cIndex.npy')
            
            for dt1,lbl,lblgt in zip(dta1,out.cpu().numpy(),gt.cpu().numpy()) :
                #print('1',lbl)
                if rl : 
                    lbl = labelMapping[int(lbl)]
                    #print('2',lbl)
                    #spamwriter.writerow([dt1,'F'+'{:04d}'.format(lbl)])
                    spamwriter.writerow([dt1,lbl])
                else : 
                    #spamwriter.writerow([dt1,'F'+'{:04d}'.format(lbl+1)])
                    spamwriter.writerow([dt1,lbl+1])
            correct+= (out==gt).sum()
            wrong+= len(gt)-(out==gt).sum() 
            print('correct', correct," wrong ",wrong," ",(i+1),"/",len(dataloaderVal))
            
            total+=len(gt)
            
    correct = correct.cpu().numpy()
    wrong = wrong.cpu().numpy()
    
    acc = truediv(correct,total)
    wacc = truediv(wrong,total)
    
    gacc +=correct 
    glen +=total
    
    print('ACC VAL '+fileName+'-'+fileRName+' :  ',acc,' wr : ',wacc)
    
    ofile.close()
    print('grand acc '+dirName+'-'+fileName+'-'+fileRName+' :  ',truediv(gacc,glen))


########################################################################################


typeTrain = args.typeTrain

test = True
if test : 
    validatePre(modelType=1,useBCE =True,fr =0, wh = 1,wcl = 1, nmlz = 1, wvl = 0,includeVal =True, dirName = 'models/',use_best=True)
    exit(0)