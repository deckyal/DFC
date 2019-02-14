import numpy as np
import file_walker
import re
import cv2
from operator import truediv
import matplotlib.pyplot as plt
from pathlib import Path
#import tensorflow as tf
import random
from config import *
from scipy.integrate.quadrature import simps
import math
from scipy.stats import multivariate_normal
import os
from random import randint
import glob
from scipy.integrate import simps
from PIL import Image,ImageFilter

def checkDirMake(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def read_kp_file(filename):
     
    x = []
    
    if ('pts' in filename) :
        with open(filename) as file:
            data2 = [re.split(r'\t+',l.strip()) for l in file]
        for i in range(len(data2)) :
            if(i not in [0,1,2,len(data2)-1]):
                x.append([ float(j) for j in data2[i][0].split()] )
    return np.asarray(x)



def make_heatmap(image_name,t_image,add,y_batch,isRandom = True,percent_heatmap = .1,percent_heatmap_e = .05):
    
    tBase = os.path.basename(image_name)
    tName,tExt = os.path.splitext(tBase)
    theDir =  os.path.dirname(image_name)+"/../heatmap-"+add+"/"
    
    if not os.path.exists(theDir):
        os.makedirs(theDir)
        
    fName =theDir+tName+".npy"
    
    #print(fName)
    try : 
        b_channel,g_channel,r_channel = t_image[:,:,0],t_image[:,:,1],t_image[:,:,2]
    except : 
        print(image_name)
    
    if os.path.isfile(fName) and isRandom: 
        newChannel = np.load(fName)
        print("using saved npy")
    else :    
        print("make npy "+add)
        newChannel = b_channel.copy(); newChannel[:] = 0
        y_t = y_batch
        
        if isRandom : 
            t0,t1,t2,t3 = get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False,
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ))
        else : 
            t0,t1,t2,t3 = get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False)
        #print(t0,t1,t2,t3)
        
        l_cd,rv = get_list_heatmap(0,None,t2-t0,t3-t1,percent_heatmap)
        l_cd_e,rv_e = get_list_heatmap(0,None,t2-t0,t3-t1,percent_heatmap_e)
        
        height, width,_ = t_image.shape
        
        scaler = 255/np.max(rv)
        #addOne = randint(0,2),addTwo = randint(0,2)
        for iter in range(68) :
            #print(height,width)
            if random: 
                ix,iy = int(y_t[iter]),int(y_t[iter+68])
            else : 
                ix,iy = int(y_t[iter])+randint(0,2),int(y_t[iter+68])+randint(0,2)
            #Now drawing given the center
            if iter in range(36,48): 
                l_cd_t = l_cd_e
                rv_t = rv_e
            else : 
                l_cd_t = l_cd
                rv_t = rv
            
            for iter2 in range(len(l_cd_t)) : 
                value = int(rv_t[iter2]*scaler)
                if newChannel[inBound(iy+l_cd_t[iter2][0],0,height-1), inBound(ix + l_cd_t[iter2][1],0,width-1)] < value : 
                    newChannel[inBound(iy+l_cd_t[iter2][0],0,height-1), inBound(ix + l_cd_t[iter2][1],0,width-1)] = int(rv_t[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
        
        #np.save(fName,newChannel)
    
    return newChannel

def get_enlarged_bb(the_kp = None, div_x = 2, div_y = 2, images = None,is_bb = False, displacement = 0, displacementxy = None):
    
    if not is_bb : 
        if displacementxy is not None : 
            t = get_bb(x_list = the_kp[:68],y_list = the_kp[68:],
                       adding_xmin=displacementxy,adding_xmax=displacementxy,
                       adding_ymin=displacementxy,adding_ymax=displacementxy)
        else : 
            t = get_bb(x_list = the_kp[:68],y_list = the_kp[68:],adding  = displacement)
    else : 
        t = the_kp
                
    l_x = (t[2]-t[0])/div_x
    l_y = (t[3]-t[1])/div_y
    
    x1 = int(max(t[0] - l_x,0))
    y1 = int(max(t[1] - l_y,0))
    
    x_min = x1; y_min = y1;
    
    #print tImage.shape
    x2 = int(min(t[2] + l_x,images.shape[1]))
    y2 = int(min(t[3] + l_y,images.shape[0]))
    
    return t,l_x,l_y,x1,y1,x_min,y_min,x2,y2

def inBoundN(input,min,max):
    if input < min : 
        return min 
    elif input > max : 
        return max 
    return input

def inBound(input,min,max):
    if input < min : 
        return int(min) 
    elif input > max : 
        return int(max) 
    return int(input)

def inBound_tf(input,min,max):
    if input < min : 
        return int(min) 
    elif input > max : 
        return int(max) 
    return int(input)

def eval(input):
    if input < 0 : 
        return 0
    else :
        return input
    
def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars)) 


def addPadd(im) : 
    #im = cv2.imread("./test-frontal.png")
    height, width, channels =im.shape
    desired_size = np.max(np.array([height,width]))
    
    add_x,add_y = 0,0
    
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    if height > width : #so shift x 
        add_x = left
    else:
        add_y = top 
        
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    #print top,bottom,left,right
    '''cv2.imshow("image", new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return new_im,add_x,add_y

#assumes p_a and p_b are both positive numbers that sum to 100
def myRand(a, p_a, b, p_b):
    return a if random.uniform(0,100) < p_a else b 



def get_list_heatmap(center,cov,image_size_x,image_size_y,percent_radius,exact_radius = None) :
    
    radius_x = int(image_size_x * percent_radius)
    radius_y = int(image_size_y * percent_radius)
    
    #print(radius_x,radius_y)
    
    l_cd = []
    
    t_radius_x = radius_x
    t_radius_y = radius_y
    
    if t_radius_x <= 0 : 
        t_radius_x = 1
    if t_radius_y <= 0 : 
        t_radius_y = 1 
        
    
    if exact_radius is not None : 
        t_radius_x = cov
        t_radius_y = cov
        
    #print(t_radius_x,t_radius_y,"radius")
    
    for x in range(center-t_radius_x,center+t_radius_x) :
        '''print((center-x)/t_radius_y)
        print(math.acos((center-x)/t_radius_y))    
        print(math.sin(math.acos((center-x)/t_radius_y)))'''
        
        yspan = t_radius_y*math.sin(math.acos(inBoundN((center-x)/t_radius_y,-1,1)));
        for y in range (int(center-yspan),int(center+yspan))  : 
            l_cd.append([x,y])
            
    l_cd = np.asarray(l_cd)
    
    mean = [center,center]
    
    if cov is None : 
        rv = multivariate_normal.pdf(l_cd,mean = mean, cov = [t_radius_x,t_radius_y])
    else :
        rv = multivariate_normal.pdf(l_cd,mean = mean, cov = [cov,cov])
        
    return l_cd,rv
    
def get_bb(x_list, y_list, length = 68,swap = False,adding = 0,adding_xmin=None, adding_xmax = None,adding_ymin = None, adding_ymax = None,show=False):
    #print x_list,y_list
    xMin = 999999;xMax = -9999999;yMin = 9999999;yMax = -99999999;
    
    if show : 
        print(x_list, y_list)
        
    for i in range(length): #x
        if xMin > x_list[i]: 
            xMin = int(x_list[i])
        if xMax < x_list[i]: 
            xMax = int(x_list[i])
    
        if yMin > y_list[i]:
            yMin = int(y_list[i])
        if yMax < y_list[i]: 
            yMax = int(y_list[i])
 
        #if show : 
        #    print("ymin : ",yMin,'ymax : ',yMax)
    l_x = xMax - xMin
    l_y = yMax - yMin
    #print(xMin,xMax,yMin,yMax)
    if swap : 
        return [xMin,xMax,yMin,yMax]
    else : 
        if adding_xmin is None: 
            if show : 
                print("return ",[xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y])
            return [xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y]
        else :
            return [xMin+adding_xmin*l_x,yMin+adding_ymin*l_y,xMax+adding_xmax*l_x,yMax+adding_ymax*l_y] 


def get_bb_face(seq_size=2,synthetic = False,path= "images/bb/"):
    
    list_gt = []
    list_labels = []
    list_labels_t = []
    
    for f in file_walker.walk(curDir +path):
        #print(f.name, f.full_path) # Name is without extension
        if f.isDirectory: # Check if object is directory
            for sub_f in f.walk():
                if sub_f.isFile:
                    if('txt' in sub_f.full_path): 
                        #print(sub_f.name, sub_f.full_path) #this is the groundtruth
                        list_labels_t.append(sub_f.full_path)
                if sub_f.isDirectory: # Check if object is directory
                    list_img = []
                    for sub_sub_f in sub_f.walk(): #this is the image
                        list_img.append(sub_sub_f.full_path)
                    list_gt.append(sorted(list_img))
    
    list_gt = sorted(list_gt)
    list_labels_t = sorted(list_labels_t)
    
    
    for lbl in list_labels_t : 
        
        with open(lbl) as file:
            x = [re.split(r',+',l.strip()) for l in file]
        y = [ list(map(int, i)) for i in x]
        list_labels.append(y)
    
    
    if seq_size is not None : 
        list_images = []
        list_ground_truth = []
        for i in range(0,len(list_gt)): 
            counter = 0
            for j in range(0,int(len(list_gt[i])/seq_size)):
                
                temp = []
                temp2 = []
                for z in range(counter,counter+seq_size):
                    temp.append(list_gt[i][z])
                    #temp2.append([list_labels[i][z][2],list_labels[i][z][3],list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                    if not synthetic : 
                        temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                    else : 
                        #temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                        temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                
                counter+=seq_size
                #print counter
                    
                list_images.append(temp) 
                list_ground_truth.append(temp2) 
    else : 
        list_images = []
        list_ground_truth = []
        for i in range(0,len(list_gt)): #per folder 
            temp = []
            temp2 = []
            for j in range(0,len(list_gt[i])):#per number of seq * number of data/seq_siz 
                
                temp.append(list_gt[i][j])
                #temp2.append([list_labels[i][z][2],list_labels[i][z][3],list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                if not synthetic : 
                    temp2.append([list_labels[i][j][0],list_labels[i][j][1],list_labels[i][j][0]+list_labels[i][j][2],list_labels[i][j][1]+list_labels[i][j][3]])
                else : 
                    #temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                    temp2.append([list_labels[i][j][0],list_labels[i][j][1],list_labels[i][j][2],list_labels[i][j][3]])
            
                    
            list_images.append(temp) 
            list_ground_truth.append(temp2)
        
    '''
    print len(list_images)
    print len(list_ground_truth)
            
    print (list_images[0])
    print (list_ground_truth[0])

    img = cv2.imread(list_images[0][0])
    
    cv2.rectangle(img,(list_ground_truth[0][0][2],list_ground_truth[0][0][3]),(list_ground_truth[0][0][4],list_ground_truth[0][0][5]),(255,0,255),1)
    cv2.imshow('jim',img)
    cv2.waitKey(0)
    '''
    return[list_images,list_ground_truth]#2d list of allsize, seqlength, (1 for image,6 for bb)

