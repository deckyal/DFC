'''
Created on Nov 22, 2018

@author: deckyal
'''

from operator import truediv

from face_localiser import face_localiser#
from lib_yolo.model import FaceDetectionRegressor#

import torch
import torch.nn as nn

import tensorflow as tf#

import cv2
import numpy as np
internal_c = 32
nc = 3 

from torch.autograd import Variable

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()
        
    def forward(self, x,useBCE=True):
        out = self.linear1(x)
        out = self.linear2(out)
        if useBCE : 
            return self.sigmoid(out)
        else : 
            return out
        #return out

class VAERegression(nn.Module):
    def __init__(self, input_size, n_z):
        super(VAERegression, self).__init__()
        self.linearMean = nn.Linear(input_size, n_z)
        self.linearStd = nn.Linear(input_size, n_z)
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()
        
    def forward(self, x):
        mu = self.linearMean(x)
        logvar = self.linearStd(x)
        
        z = reparametrize(mu, logvar)
        
        return z,mu,logvar
        #return out
        
        
class LogisticRegressionShort(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionShort, self).__init__()
        self.linear1 = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(.5)
    
    def forward(self, x,useBCE=True,useDropout = False):
        if useDropout : 
            out = self.linear1(self.dropout(x))
        else : 
            out = self.linear1(x)
            
        if useBCE : 
            return self.sigmoid(out)
        else : 
            return out
        return out
    
def reparametrize(mu,logvar):
    std = logvar.mul(.5).exp_()
    if True : #self.have_cuda : 
        eps = torch.cuda.FloatTensor(std.size()).normal_()
    else : 
        eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu) 
    

class LogisticRegressionC(nn.Module):
    def __init__(self, input_size,input_size2, num_classes):
        super(LogisticRegressionC, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256+input_size2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()
    
    def forward(self, x,x2,useBCE=True):
        out = self.linear1(x)
        out = self.linear2(torch.cat((out,x2),1))
        if useBCE : 
            return self.sigmoid(out)
        else : 
            return out
        #return out #self.sigmoid(out)
    


class DAEE(nn.Module):
    def __init__(self,debug = False):
        super(DAEE,self).__init__()
        
        self.skip = True
        self.maxpool = nn.MaxPool2d(2)
        self.debug = debug
        
        #Input is (nc) x 64 x 64
        
        self.conv0 = nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias = False)
        
        self.conv1 = nn.Conv2d(nc,internal_c,kernel_size=3,stride=1,padding=1,bias = False)
        self.relrlu = nn.LeakyReLU(.2, inplace=True)
        
        #State size. (ndf)*32*32
        self.conv2 = nn.Conv2d(32, 48, 3,1,1,bias = False)
        
        #State size. (ndf*2)*16*16
        self.conv3 = nn.Conv2d(48, 64, 3,1,1,bias = False)
        
        #State size. (ndf*4)*8*8
        self.conv4 = nn.Conv2d(64, 72, 3,1,1,bias = False)
        
        #State size. (ndf*8)*4*4
        self.conv5 = nn.Conv2d(72, 108, 3,1,1,bias = False) 
        self.sigmoid = nn.Sigmoid()
        
        #self.fcconv = nn.Linear(1024,hidden_size)
        #self.fclabel = nn.Linear(8,hidden_size)
        #self.bl = nn.Linear(896,128)
    
        self.fce = nn.Linear(5292,512)
        
        #############################
        
        self.fcd1 = nn.Linear(512,5292)
         
        self.nnUpsample = nn.Upsample(scale_factor = 2)
        
        self.dconv1 = nn.ConvTranspose2d(in_channels = 108,out_channels = 72,kernel_size=3, stride = 1,padding = 1)
        self.relu =  nn.ReLU(True)
        
        #PrintLayer(),
        
        #State size. (ngf*8)*4*4
        self.dconv2 = nn.ConvTranspose2d(in_channels =72,out_channels = 64,kernel_size=3, stride = 1,padding = 1)
        #State size. (ngf*4)*8*8
        
        self.dconv3 = nn.ConvTranspose2d(in_channels = 64,out_channels = 48,kernel_size=3, stride = 1,padding = 1)
        
        #State size. (ngf*2)*16*16
        self.dconv4 = nn.ConvTranspose2d(in_channels = 48,out_channels = 32,kernel_size=3, stride = 1,padding = 1)
        
        #State size. (ngf)*32*32
        self.dconv5 = nn.ConvTranspose2d(in_channels = 32,out_channels = 3,kernel_size=3, stride = 1,padding = 1)
        
        self.dconv6 = nn.ConvTranspose2d(in_channels = 3,out_channels = 3,kernel_size=3, stride = 1,padding = 1)
        
        self.tanh = nn.Tanh()
        #State size. (nc)*64*64
        
        #self.fclabel = nn.Linear(8,hidden_size)
        
        #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1
        
    
    def forward(self,input):
        
        if self.debug : 
            print("Input shape : ",input.shape)
        xc0 = self.relrlu(self.conv0(input))
        
        if self.debug : 
            print("Input shape : ",xc0.shape)
        #xc1 = self.relrlu(self.maxpool(self.conv1b(self.conv1(xc0))))
        xc1 = self.relrlu(self.maxpool(self.conv1(xc0)))
        if self.debug : 
            print("e Shape 1 : ",xc1.shape)
        xc2 = self.relrlu(self.maxpool(self.conv2(xc1)))
        if self.debug : 
            print("e Shape 2 : ",xc2.shape)
        xc3 = self.relrlu(self.maxpool(self.conv3(xc2)))
        if self.debug : 
            print("conv3",xc3.shape)
        xc4 = self.relrlu(self.maxpool(self.conv4(xc3)))
        if self.debug : 
            print("conv4",xc4.shape)
        xc5 = self.relrlu(self.maxpool(self.conv5(xc4)))
        if self.debug : 
            print("conv5",xc5.shape)
        
        xe = xc5.view(xc5.size(0),-1)
        #print(xe.shape)
        xe = self.fce(xe)
        
        #x = self.fcconv(x.view(x.size(0),-1))
        #x = self.bl(x)
        xd = self.fcd1(xe).view(xe.size(0),108,7,7)
        
        if self.debug : 
            print("Genereator : ",xd.shape)
        xd1 = self.relu(self.nnUpsample(self.dconv1(xd)))
        if self.debug : 
            print("dconv1b,dconv1bb, xc4 : ",xd1.shape,self.dconv1(xd).shape,xc4.shape)
        xd2 = self.relu(self.nnUpsample(self.dconv2(xd1+ xc4)))
        if self.debug : 
            print("dconv2b : ",xd2.shape)
        xd3 = self.relu(self.nnUpsample(self.dconv3(xd2+ xc3)))
        if self.debug : 
            print("dconv3b : ",xd3.shape)
        xd4 = self.relu(self.nnUpsample(self.dconv4(xd3+ xc2)))
        if self.debug : 
            print("dconv4b : ",xd4.shape)
        xd5 = self.relu(self.nnUpsample(self.dconv5(xd4+ xc1)))
        if self.debug : 
            print("dconv5 : ",xd5.shape)
        xd6 = self.tanh(self.dconv6(xd5+ xc0)) 
        if self.debug : 
            print("dconv6 : ",xd6.shape)
            
        return xd6,xe



class DAEEH(nn.Module):
    def __init__(self,debug = False):
        super(DAEEH,self).__init__()
        
        self.skip = True
        self.maxpool = nn.MaxPool2d(2)
        self.debug = debug
        
        #Input is (nc) x 64 x 64
        
        self.conv0 = nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias = False)
        
        self.conv1 = nn.Conv2d(nc,internal_c,kernel_size=3,stride=1,padding=1,bias = False)
        self.relrlu = nn.LeakyReLU(.2, inplace=True)
        
        #State size. (ndf)*32*32
        self.conv2 = nn.Conv2d(32, 48, 3,1,1,bias = False)
        
        #State size. (ndf*2)*16*16
        self.conv3 = nn.Conv2d(48, 64, 3,1,1,bias = False)
        
        #State size. (ndf*4)*8*8
        self.conv4 = nn.Conv2d(64, 72, 3,1,1,bias = False)
        
        #State size. (ndf*8)*4*4
        self.conv5 = nn.Conv2d(72, 108, 3,1,1,bias = False) 
        self.sigmoid = nn.Sigmoid()
        
        #self.fcconv = nn.Linear(1024,hidden_size)
        #self.fclabel = nn.Linear(8,hidden_size)
        #self.bl = nn.Linear(896,128)
    
        self.fce = nn.Linear(5292,512)
        
    def forward(self,input):
        
        if self.debug : 
            print("Input shape : ",input.shape)
        xc0 = self.relrlu(self.conv0(input))
        
        if self.debug : 
            print("Input shape : ",xc0.shape)
        #xc1 = self.relrlu(self.maxpool(self.conv1b(self.conv1(xc0))))
        xc1 = self.relrlu(self.maxpool(self.conv1(xc0)))
        if self.debug : 
            print("e Shape 1 : ",xc1.shape)
        xc2 = self.relrlu(self.maxpool(self.conv2(xc1)))
        if self.debug : 
            print("e Shape 2 : ",xc2.shape)
        xc3 = self.relrlu(self.maxpool(self.conv3(xc2)))
        if self.debug : 
            print("conv3",xc3.shape)
        xc4 = self.relrlu(self.maxpool(self.conv4(xc3)))
        if self.debug : 
            print("conv4",xc4.shape)
        xc5 = self.relrlu(self.maxpool(self.conv5(xc4)))
        if self.debug : 
            print("conv5",xc5.shape)
        
        xe = xc5.view(xc5.size(0),-1)
        #print(xe.shape)
        xe = self.fce(xe)
            
        return xe





class GeneralDAE():
    
    def __init__(self,preload = True, cuda = True,type = 1):
        
        if type == 1 : 
            self.fileNameList = ["1st_model/AESE_WB_WE_3x3_224",'1st_model/AESE_WB_WE_3x3_224_DownScale','1st_model/AESE_WB_WE_3x3_224_GBlur','1st_model/AESE_WB_WE_3x3_224_GNoise','1st_model/AESE_WB_WE_3x3_224_CScale']
        else : 
            self.fileNameList = ["2nd_model/AESE_WB_WE_3x3_224",'2nd_model/AESE_WB_WE_3x3_224_DownScale','2nd_model/AESE_WB_WE_3x3_224_GBlur','2nd_model/AESE_WB_WE_3x3_224_GNoise','2nd_model/AESE_WB_WE_3x3_224_CScale']
        self.preload = True 
        self.cuda = True
        
        self.DAE1 = DAEE()
        self.DAE2 = DAEE()
        self.DAE3 = DAEE()
        self.DAE4 = DAEE()
        
        if self.preload : 
            self.DAE1.load_state_dict(torch.load(self.fileNameList[1]+".pt"))
            self.DAE2.load_state_dict(torch.load(self.fileNameList[2]+".pt"))
            self.DAE3.load_state_dict(torch.load(self.fileNameList[3]+".pt"))
            self.DAE4.load_state_dict(torch.load(self.fileNameList[4]+".pt"))
            
            if self.cuda  : 
                self.DAE1 = self.DAE1.cuda()
                self.DAE2 = self.DAE2.cuda()
                self.DAE3 = self.DAE3.cuda()
                self.DAE4 = self.DAE4.cuda()
            
            self.DAE1.eval()
            self.DAE2.eval()
            self.DAE3.eval()
            self.DAE4.eval()
            
    def forward(self,inputImage,noiseType):
            
        if self.preload : 
            if (noiseType == 1): 
                return self.DAE1(inputImage)[0].detach().cpu()
            elif (noiseType == 2): 
                return self.DAE2(inputImage)[0].detach().cpu()
            elif (noiseType == 3): 
                return self.DAE3(inputImage)[0].detach().cpu()
            elif (noiseType == 4): 
                return self.DAE4(inputImage)[0].detach().cpu()
            else : 
                return inputImage.cpu()


class DVAEHalf(nn.Module):
    
    nz = 128

    def __init__(self,debug = False):
        super(DVAEHalf,self).__init__()
        
        self.fce = nn.Linear(5292,512)
        self.fcem = nn.Linear(512,self.nz)
        self.fcev = nn.Linear(512,self.nz)
    
        #############################
        
        #Input is Z, going into a convolution 
        
        self.fcd1 = nn.Linear(self.nz,5292)
        
        self.nnUpsample = nn.Upsample(scale_factor = 2)
        
        self.dconv1 = nn.ConvTranspose2d(in_channels = 108,out_channels = 72,kernel_size=3, stride = 1,padding = 1)
        self.relu =  nn.ReLU(True)
        
        #PrintLayer(),
        
        #State size. (ngf*8)*4*4
        self.dconv2 = nn.ConvTranspose2d(in_channels =72,out_channels = 64,kernel_size=3, stride = 1,padding = 1)
        #State size. (ngf*4)*8*8
        
        self.dconv3 = nn.ConvTranspose2d(in_channels = 64,out_channels = 48,kernel_size=3, stride = 1,padding = 1)
        
        #State size. (ngf*2)*16*16
        self.dconv4 = nn.ConvTranspose2d(in_channels = 48,out_channels = 32,kernel_size=3, stride = 1,padding = 1)
        
        #State size. (ngf)*32*32
        self.dconv5 = nn.ConvTranspose2d(in_channels = 32,out_channels = 3,kernel_size=3, stride = 1,padding = 1)
        
        self.dconv6 = nn.ConvTranspose2d(in_channels = 3,out_channels = 3,kernel_size=3, stride = 1,padding = 1)
        
        self.tanh = nn.Tanh()
        #State size. (nc)*64*64
        #self.fclabel = nn.Linear(8,hidden_size)
        #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1
    
    def reparametrize(self, mu,logvar):
        std = logvar.mul(.5).exp_()
        if True : #self.have_cuda : 
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else : 
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def forward(self,input):
        
        xe = self.fce(xc5.view(xc5.size(0),-1))
        
        mu,logvar = self.fcem(xe),self.fcev(xe)
        z = self.reparametrize(mu, logvar)
        
        xc5 = self.relu(self.fcd1(z)).view(xc5.size(0),108,7,7)
        
        if self.debug : 
            print("Genereator : ",xc5.shape)
        xd1 = self.relu(self.nnUpsample(self.dconv1(xc5)))
        if self.debug : 
            print("dconv1b, xc4 : ",xd1.shape)
        xd2 = self.relu(self.nnUpsample(self.dconv2(xd1)))
        if self.debug : 
            print("dconv2b : ",xd2.shape)
        xd3 = self.relu(self.nnUpsample(self.dconv3(xd2)))
        if self.debug : 
            print("dconv3b : ",xd3.shape)
        xd4 = self.relu(self.nnUpsample(self.dconv4(xd3)))
        if self.debug : 
            print("dconv4b : ",xd4.shape)
        xd5 = self.relu(self.nnUpsample(self.dconv5(xd4)))
        if self.debug : 
            print("dconv5 : ",xd5.shape)
        xd6 = self.tanh(self.dconv6(xd5)) 
        if self.debug : 
            print("dconv6 : ",xd6.shape)
            
        return xd6,mu,logvar


class FacialLocaliser():
    
    from face_localiser import face_localiser
    from lib_yolo.model import FaceDetectionRegressor

    import tensorflow as tf
    def __init__(self,is3D = False,modelDir='./models/',gpu_fracts = .125):
        
        self.image_size = 128
        self.is3D = is3D
        self.name_save = "dt-inception"
        if self.is3D : 
            self.name_save += "-3D"
        self.channels = 3
        
        
        self.f = face_localiser(self.image_size,False,self.channels)
        self.x,self.y,self.pred = self.f.build()
        
        self.saver = tf.train.Saver(max_to_keep=2,save_relative_paths=True)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fracts
        config.gpu_options.visible_device_list = "0"
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        
        print("using : ",self.name_save)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(modelDir+self.name_save))
        
        # FaceDetectionRegressor
        self.model = FaceDetectionRegressor()
        # load weights#
        self.model.load_weights(modelDir+'yolo')
        print('test')
        
    def forward(self,img,showResult = False):
        #print('forwardking')
        #It's assumed that the image is cropped in the center.
        tImage = cv2.imread(img)
        
        predictions = self.model.predict(tImage, merge=True)
        #print('forwardking')
        if len(predictions) > 0 : 
            for box in predictions :
                #print(box)
                t = np.zeros(4)
                t[0]  = int((box['x'] - box['w'] / 2.))#left
                t[2]  = int((box['x'] + box['w'] / 2.))#right 
                t[1]= int((box['y'] - box['h'] / 2.)) #top 
                t[3] = int((box['y'] + box['h'] / 2.)) #bot
                
                
                l_x = (t[2]-t[0])/2 
                l_y = (t[3]-t[1])/2  
                
                x1 = int(max(t[0] - l_x,0))
                y1 = int(max(t[1] - l_y,0))
                x1a = int(t[0] - l_x)
                y1a = int(t[1] - l_y)
                
                #print tImage.shape
                x2 = int(min(t[2] + l_x,tImage.shape[1]))
                y2 = int(min(t[3] + l_y,tImage.shape[0]))
                x2a = int(t[2] + l_x)
                y2a = int(t[3] + l_y)
                
                tIm = np.zeros((y2a-y1a,x2a-x1a,3))
                
                tImage = tImage[int(y1):int(y2),int(x1):int(x2)].copy();#cv2.rectangle(tImage,(left, top), (right, bot),(0,255,0),3)
                
                dx_min,dx_max,dy_min,dy_max = 0,0,0,0
                if x1a < 0 : 
                    dx_min = -x1a
                    
                if x2a > tImage.shape[1]: 
                    dx_max = x2a - tImage.shape[1]
                    
                if y1a < 0 : 
                    dy_min = -y1a 
                    
                if y2a > tImage.shape[2]: 
                    dy_max = y2a -tImage.shape[2]
                
                #print(dy_min,dy_max,dx_min,dx_max)
                #print(tImage.shape,tIm.shape)
                
                #print(tIm[0+dy_min:(y2-y1)+dy_min,0+dx_min :(x2-x1)+dx_min].shape)
                tIm[0+dy_min:(y2-y1)+dy_min,0+dx_min :(x2-x1)+dx_min] = tImage
                
        #print('forwardking')
        #print(tImage.shape)
        height, width, channels = tImage.shape
        ratioHeightR =truediv(height,self.image_size)
        ratioWidthR =truediv(width,self.image_size)
        #print(ratioHeightR,ratioWidthR)
        r_image = cv2.resize(tImage, (self.image_size,self.image_size))
        #cv2.imshow('test',r_image)
        #cv2.waitKey(0)
                
        predicted = self.pred.eval(feed_dict = {self.x:np.expand_dims(r_image, axis=0)},session = self.sess)[0]
        #print(predicted)
        #Now recovering from the resized image to original image size
        
        predicted[0:68]*=ratioWidthR
        predicted[68:136]*=ratioHeightR 
        
        if showResult : 
            tImage2 = r_image
            x_list = predicted[0:68]
            y_list = predicted[68:136]
        
            bb = self.get_bb(x_list,y_list)
            
            cv2.rectangle(tImage2,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
            for i in range(68) :
                cv2.circle(tImage2,(int(x_list[i]),int(y_list[i])),3,(0,255,0))
            
            
            predicted[0:68]*=ratioWidthR
            predicted[68:136]*=ratioHeightR 
            
            x_list = predicted[0:68]
            y_list = predicted[68:136]
        
            bb = self.get_bb(x_list,y_list)
            
            cv2.rectangle(tImage,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
            for i in range(68) :
                cv2.circle(tImage,(int(x_list[i]),int(y_list[i])),3,(0,255,0))
            
            cv2.imshow('result',tImage)
            cv2.waitKey(1)
            #cv2.imwrite('result.jpg',tImage)
            #print("done")
        return predicted
    
    
    def get_bb(self,x_list, y_list, length = 68,swap = False,adding = 0,adding_xmin=None, adding_xmax = None,adding_ymin = None, adding_ymax = None):
        #print x_list,y_list
        xMin = 999999;xMax = -9999999;yMin = 9999999;yMax = -99999999;
        
        for i in range(length): #x
            if xMin > x_list[i]: 
                xMin = int(x_list[i])
            if xMax < x_list[i]: 
                xMax = int(x_list[i])
        
            if yMin > y_list[i]: 
                yMin = int(y_list[i])
            if yMax < y_list[i]: 
                yMax = int(y_list[i])
        
        l_x = xMax - xMin
        l_y = yMax - yMin
        
        if swap : 
            return [xMin,xMax,yMin,yMax]
        else : 
            if adding_xmin is None: 
                return [xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y]
            else :
                return [xMin+adding_xmin*l_x,yMin+adding_ymin*l_y,xMax+adding_xmax*l_x,yMax+adding_ymax*l_y] 
