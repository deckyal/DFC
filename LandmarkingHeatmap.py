'''
Created on Nov 22, 2018

@author: deckyal
'''

from __future__ import print_function
import file_walker
import cv2
import os
from utils import checkDirMake
import numpy as np
import utils


def landmarkHeatmap():
    
    tDir = "/home/deckyal/eclipse-workspace/DeepModel/src/KV/data/images/FIDs"
    tDirL = "/home/deckyal/eclipse-workspace/DeepModel/src/KV/data/landmarks/FIDs/"
    tDirH = "/home/deckyal/eclipse-workspace/DeepModel/src/KV/data/heatmaps/FIDs/"
    
    theMiddleName = "FIDs"
    
    '''for f in file_walker.walk(curDir + "images/"+data+"/"):
                    if f.isDirectory: # Check if object is directory
                        #print((f.name, f.full_path)) # Name is without extension
                        for sub_f in f.walk():
                            if sub_f.isDirectory: # Check if object is directory
                                list_dta = []
                                #print sub_f.name
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                
                                if(sub_f.name == annot_name) : #If that's annot, add to labels_t 
                                    list_labels_t.append(sorted(list_dta))
                                elif(sub_f.name == 'img'): #Else it is the image
                                    list_gt.append(sorted(list_dta))
                                    counter_image+=len(list_dta)
    '''
    listData = []
    for f in file_walker.walk(tDir):#FID
        print((f.name, f.full_path)) # Name is without extension
        if f.isDirectory: 
            for sub_f in f.walk(): 
                if sub_f.isDirectory:#F00!
                    for sub_sub_f in sub_f.walk(): #MID1
                        listData.append(sub_sub_f.full_path)
                        print(sub_sub_f.full_path)
                        
                        
    from NetworkModels import FacialLocaliser
    f = FacialLocaliser()
    #points = f.forward('P00001_face2.jpg',showResult=True)
    
    for image_path in listData: 
        
        res = f.forward(image_path,showResult = False)
        
        filePath = image_path.split(os.sep)
        ifolder = filePath.index(theMiddleName)
        
        image_name = filePath[-1]
        
        annot_name_L = os.path.splitext(image_name)[0]+'.pts'
        annot_name_H = os.path.splitext(image_name)[0]+'.npy'
        
        sDirName = filePath[ifolder+1:-1]
        
        dLandmarks = tDirL+'/'.join(sDirName)
        dHeatmaps = tDirH+'/'.join(sDirName)
        
        checkDirMake(dLandmarks)
        checkDirMake(dHeatmaps)
        
        #print(filePath,sDirName)
        #print(dLandmarks,dHeatmaps)
        
        finalTargetL = dLandmarks+'/'+annot_name_L
        finalTargetH = dHeatmaps+'/'+annot_name_H
        
        #making the landmark
        file = open(finalTargetL,'w')
        file.write('version: 1\n')
        file.write('n_points: 68\n')
        file.write('{\n')
        for i in range(68) :
            file.write(str(res[i])+' '+str(res[i+68])+'\n')
        file.write('}')
        file.close()
        
        #Making the heatmap
        
        tImage = cv2.imread(image_path)
        
        b_channel,g_channel,r_channel = tImage[:,:,0],tImage[:,:,1],tImage[:,:,2]
        newChannel = b_channel.copy(); newChannel[:] = 0
        
        t0,t1,t2,t3 = utils.get_bb(res[0:68], res[68:])
        
        l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
        height, width,_ = tImage.shape
        
        wx = t2-t0
        wy = t3-t1
    
        scaler = 255/np.max(rv)
        
        for iter in range(68) :
            ix,iy = int(res[iter]),int(res[iter+68])
            
            #Now drawing given the center
            for iter2 in range(len(l_cd)) : 
                value = int(rv[iter2]*scaler)
                if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                    newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
        
        '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
        cv2.imshow("combined",tImage2)
        cv2.waitKey(0)'''
        
        np.save(finalTargetH,newChannel)
        
        print(finalTargetL,'--',finalTargetH)
        
        doTest = True
        if doTest :
            #now test reread 
            tImage = cv2.imread(image_path)
            ldmarks = np.asarray(utils.read_kp_file(finalTargetL)).flatten('F')
            htmaps = np.load(finalTargetH)
            
            b_channel,g_channel,r_channel = tImage[:,:,0],tImage[:,:,1],tImage[:,:,2]
            tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
            
            for i in range(68) :
                cv2.circle(tImage2,(int(ldmarks[i]),int(ldmarks[i+68])),3,(0,255,0))
            
            cv2.imshow('tst',tImage2)
            cv2.waitKey(1)
            
        
        #exit(0)
        '''if isVideo :  
            middle = filePath[ifolder+2:-2]
            print(middle)
            middle = '/'.join(middle)
            
            finalTargetPathI = targetDir+middle+'/img/'
            finalTargetPathA = targetDir+middle+'/annot/'
        else : 
            finalTargetPathI = targetDir+'img/'
            finalTargetPathA = targetDir+'annot/'
        '''
    
    

def clean():
    
    theMiddleName = "FIDs"
    
    cleanerNames = ['G','GC','GCAE']
    cId = 2
    
    newFolderName = "./images-"+cleanerNames[cId]+"/FIDs/"
    tDir = "./images/FIDs"
    tDirL = "./landmarks-"+cleanerNames[cId]+"/FIDs/"
    tDirH = "./heatmaps-"+cleanerNames[cId]+"/FIDs/"
    
    
    checkDirMake(newFolderName)

    
    listData = []
    for f in file_walker.walk(tDir):#FID
        print((f.name, f.full_path)) # Name is without extension
        if f.isDirectory: 
            for sub_f in f.walk(): 
                if sub_f.isDirectory:#F00!
                    for sub_sub_f in sub_f.walk(): #MID1
                        listData.append(sub_sub_f.full_path)
                        print(sub_sub_f.full_path)
    
    
    
                        
    from NetworkModels import FacialLocaliser, DAEE, LogisticRegression, DAEEH,\
    GeneralDAE
    f = FacialLocaliser()
    
    
    ###########Network definitions############3
    
    import torch
    from torchvision.transforms import transforms
    from PIL import Image
    from utils import UnNormalize
    image_size = 224
    
    #Denoiser 
    
    cl_type = cId#1 is combine, 2 is half, 0 is general 
    model_dir = ['1st_model/','2nd_model/']
    
    #General Denoiser 
    
    net = DAEE()
    net.load_state_dict(torch.load(model_dir[1]+"AESE_WB_WE_3x3_224"+'.pt'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()
    
    #Multiple expert denoiser 
    
    GD = GeneralDAE(type=2)
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    if cl_type == 1 : 
        netAEC = DAEE()
        netAEC.load_state_dict(torch.load(model_dir[0]+'combineAE.pt'))
        netAEC = netAEC.cuda()
        netAEC.eval()
        
        model_lg = LogisticRegression(512, 5)
        model_lg.load_state_dict(torch.load(model_dir[0]+'combineCL.pt'))
        model_lg = model_lg.cuda()
        model_lg.eval()
    elif cl_type == 2 : 
        netAEC = DAEEH()
        netAEC.load_state_dict(torch.load(model_dir[0]+'combineAEH.pt'))
        netAEC = netAEC.cuda()
        netAEC.eval()
        
        model_lg = LogisticRegression(512, 5)
        model_lg.load_state_dict(torch.load(model_dir[0]+'combineCLH.pt'))
        model_lg = model_lg.cuda()
        model_lg.eval()

    
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #points = f.forward('P00001_face2.jpg',showResult=True)
    
    for image_path in listData: 
        
        #cv2.imshow('test',cv2.imread(image_path))
        #cv2.waitKey(0)
        
        ImageB = Image.open(image_path).resize((image_size,image_size))
        #tImageB.show()
        
        #tImageB = generalNoise(tImageB,nt,npram)
        tImageB = transform(ImageB).unsqueeze(0).cuda()
        
        if cl_type == 0 :
            res  = net.forward(tImageB)[0].detach().cpu()
        else :
            if cl_type == 1 : 
                recon_batch,xe = netAEC(tImageB)
            elif cl_type == 2 :  
                xe = netAEC(tImageB)
            labels = model_lg(xe,False)
            x, y = torch.max(labels, 1)
            res = GD.forward(tImageB, y[0])
        
        #Now saving the cleaned one 
        img_res = unorm(res[0].detach().cpu()).numpy()*255 
        img_res = img_res.transpose((1,2,0))
        img_res = cv2.cvtColor(img_res.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
        
        #cv2.imshow('test',img_res)
        #cv2.waitKey(0)
        
        
        filePath = image_path.split(os.sep)
        ifolder = filePath.index(theMiddleName)
        
        image_name = filePath[-1]
        
        annot_name_L = os.path.splitext(image_name)[0]+'.pts'
        annot_name_H = os.path.splitext(image_name)[0]+'.npy'
        
        sDirName = filePath[ifolder+1:-1]
        
        dirImage = newFolderName +'/'.join(sDirName) + '/'
        dImage = dirImage+image_name
        dLandmarks = tDirL+'/'.join(sDirName)
        dHeatmaps = tDirH+'/'.join(sDirName)
        
        checkDirMake(dirImage)
        cv2.imwrite(dImage, img_res)
        #print(sDirName,image_name,dImage)
        #exit(0)
        
        res = f.forward(dImage,showResult = False)
        
        
        checkDirMake(dLandmarks)
        checkDirMake(dHeatmaps)
        
        #print(filePath,sDirName)
        #print(dLandmarks,dHeatmaps)
        
        finalTargetL = dLandmarks+'/'+annot_name_L
        finalTargetH = dHeatmaps+'/'+annot_name_H
        
        #making the landmark
        file = open(finalTargetL,'w')
        file.write('version: 1\n')
        file.write('n_points: 68\n')
        file.write('{\n')
        for i in range(68) :
            file.write(str(res[i])+' '+str(res[i+68])+'\n')
        file.write('}')
        file.close()
        
        #Making the heatmap
        
        tImage = cv2.imread(dImage)
        
        b_channel,g_channel,r_channel = tImage[:,:,0],tImage[:,:,1],tImage[:,:,2]
        newChannel = b_channel.copy(); newChannel[:] = 0
        
        t0,t1,t2,t3 = utils.get_bb(res[0:68], res[68:])
        
        l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
        height, width,_ = tImage.shape
        
        wx = t2-t0
        wy = t3-t1
    
        scaler = 255/np.max(rv)
        
        for iter in range(68) :
            ix,iy = int(res[iter]),int(res[iter+68])
            
            #Now drawing given the center
            for iter2 in range(len(l_cd)) : 
                value = int(rv[iter2]*scaler)
                if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                    newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
        
        '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
        cv2.imshow("combined",tImage2)
        cv2.waitKey(0)'''
        
        np.save(finalTargetH,newChannel)
        
        print(finalTargetL,'--',finalTargetH)
        
        doTest = True
        if doTest :
            #now test reread 
            tImage = cv2.imread(dImage)
            ldmarks = np.asarray(utils.read_kp_file(finalTargetL)).flatten('F')
            htmaps = np.load(finalTargetH)
            
            b_channel,g_channel,r_channel = tImage[:,:,0],tImage[:,:,1],tImage[:,:,2]
            tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
            
            for i in range(68) :
                cv2.circle(tImage2,(int(ldmarks[i]),int(ldmarks[i+68])),3,(0,255,0))
            
            cv2.imshow('tst',tImage2)
            cv2.waitKey(1)
            
            
clean()


