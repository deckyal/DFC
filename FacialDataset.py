'''
Created on Oct 17, 2018

@author: deckyal
'''

from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms
import csv
import torchvision.transforms.functional as F
import numbers

import file_walker
from utils import *
from config import *
import utils

def read_kp_file(filename):
    x = []
    if ('pts' in filename) :
        with open(filename) as file:
            data2 = [re.split(r'\t+',l.strip()) for l in file]
        for i in range(len(data2)) :
            if(i not in [0,1,2,len(data2)-1]):
                x.append([ float(j) for j in data2[i][0].split()] )
                
    return np.asarray(x).flatten('F')


class FamilyDataset(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = [''],transforms = None,dirImage = None,withLandmark = False, withHeatmap=False,return_data = False,
                 useIT = False,convertGT = False, multiLabel = False,convertToOneHot  = False, dirList = ['landmarks','heatmaps'],isTest = False,
                 augment = True,rl = False,calculateWeight = False):
        
        self.aug = augment
        self.isTest = isTest
        self.tDirL = curDir+""+dirList[0]+"/FIDs/"
        self.tDirH = curDir+""+dirList[1]+"/FIDs/"
        self.theMiddleName = "FIDs"
        self.useIT = useIT
        self.cGT = convertGT
        self.mLBL = multiLabel
        self.reduce_label = rl
        self.cw = calculateWeight
        self.labelWeight = None
        
        self.toOneHot = convertToOneHot
        self.cIndex = []
        
        self.l_imgs = []
        self.l_gt = []
        self.transforms = transforms
        self.dirImage = dirImage
        self.withLandmark = withLandmark
        self.withHeatmap = withHeatmap
        self.return_data = return_data
        
        tmp = []
        if self.mLBL : 
            for x in data_list : 
                print(x)
                alist = [line.rstrip() for line in open(curDir+x)]
                for data in alist :
                    d = data.split(' ')
                    imgName = d[0];
                    if self.isTest : 
                        imgLabel = '0004'#Dummy Label
                    else : 
                        imgLabel = int(d[1][1:])
                    
                    self.l_imgs.append([imgName])
                    
                    tmp.append([int(imgLabel)])
                    self.l_gt.append([imgLabel])
            print(x)
            if 'trainVal' in x : 
                unique,counts = np.unique(np.sort(np.asarray(tmp)),return_counts = True)
                self.cIndex = unique
                #print(unique,counts)
                if self.cw  :
                    #print(np.asarray(tmp).shape)
                    '''#now normalize
                    counts = np.asarray(counts,dtype =np.float32)
                    print(counts,np.sum(counts))
                    counts/=np.sum(counts)
                    self.label_weight = np.ones(600)
                    #print(counts,'sum here ', np.sum(counts))
                    self.label_weight[:counts.shape[0]]/=(counts)#np.array([unique,counts])
                    self.label_weight/=100'''
                    
                    counts = np.asarray(counts,dtype =np.float32)
                    print(counts,np.sum(counts))
                    counts = np.sum(counts) / counts
                    self.label_weight = np.zeros(540)
                    #print(counts,'sum here ', np.sum(counts))
                    self.label_weight[:counts.shape[0]]+=(counts)#np.array([unique,counts])
                    self.label_weight/=100
                    
                    print(self.label_weight)
                    
                    np.save(curDir+'cWeight.npy',self.label_weight)
                    
                np.save(curDir+'cIndex.npy',self.cIndex)
            else : 
                self.cIndex = np.load(curDir+'cIndex.npy')
                self.label_weight = np.load(curDir+'cWeight.npy')
                
                    
            #exit(0)
            
        else : 
            #open csv put the list files and gt in the list 
            for x in data_list : 
                #print(x)
                with open(curDir+"data/"+x, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in spamreader:
                        data = row[0].split(',');
                        
                        if not 'p1' in data :
                            if self.isTest : 
                                #self.l_gt.append([data[0],data[1]])
                                id1 = data[0].split('/')[0]
                                id2 = data[1].split('/')[0]
                                if id1 == id2 : 
                                    lbl = 1
                                else : 
                                    lbl = 0
                                self.l_gt.append([0,lbl])
                                self.l_imgs.append([data[0],data[1]])
                                
                                #print(x,'-',data[0],data[1],lbl)
                            else :
                                #print(x,'-',data[0],data[1],data[2],data[3],len(self.l_imgs))
                                self.l_gt.append([data[0],data[1]])
                                self.l_imgs.append([data[2],data[3]])
                                
    def __getitem__(self,index):
        
        #print('self t data index',self.l_gt[index])
        x,label  = self.l_imgs[index],self.l_gt[index]
        #print('source',x,label)
        aImgs = []
        ldmarks = []
        heatmaps = []
        limgs = []
        
        for ix in x :
            if self.withLandmark or self.withHeatmap :
                 
                filePath = (self.dirImage+ix).split(os.sep)
                #print(filePath)
                ifolder = filePath.index(self.theMiddleName)
                
                image_name = filePath[-1]
                
                annot_name_L = os.path.splitext(image_name)[0]+'.pts'
                annot_name_H = os.path.splitext(image_name)[0]+'.npy'
                
                sDirName = filePath[ifolder+1:-1]
                
                dLandmarks = self.tDirL+'/'.join(sDirName)
                dHeatmaps = self.tDirH+'/'.join(sDirName)
                
                checkDirMake(dLandmarks)
                checkDirMake(dHeatmaps)
                
                #print(filePath,sDirName)
                #print(dLandmarks,dHeatmaps)
                
                finalTargetL = dLandmarks+'/'+annot_name_L
                finalTargetH = dHeatmaps+'/'+annot_name_H
                
                if self.withLandmark : 
                    #Get the file name 
                    ldmrk = np.asarray(utils.read_kp_file(finalTargetL)).flatten('F')
                    ldmarks.append(torch.from_numpy(ldmrk))
                
                if self.withHeatmap :
                    htmp = np.load(finalTargetH)
                    
                    '''#Manual transform.
                    if self.useIT : 
                        htmp = torch.from_numpy(htmp).float()
                        #tImage = self.transform(htmp)
                    else :
                        
                        htmp = torch.from_numpy(htmp).float().div(255) 
                        mean = self.transforms['NormalizeM'][0]
                        std = self.transforms['NormalizeM'][1]
                        
                        #print(mean,std,torch.max(htmp))
                        htmp.sub_(mean).div_(std)
                        #print(mean,std,torch.max(htmp))
                        #print(htmp)
                        #img = Image.fromarray(htmp, 'L')'''
                    
                    
                    #heatmaps.append(htmp.unsqueeze(0))
                    #test = self.transforms['train1D'](img)\
            limgs.append(ix)
            #print(self.dirImage+ix)
            tImage = Image.open(self.dirImage+ix).convert("RGB")
            
            '''if self.withHeatmap and False : 
                opencvImage = cv2.cvtColor(np.array(tImage), cv2.COLOR_RGB2BGR)
                b_channel,g_channel,r_channel = opencvImage[:,:,0],opencvImage[:,:,1],opencvImage[:,:,2]
                tImageCV = cv2.merge((b_channel, g_channel,r_channel, htmp))'''
            
            if self.transforms is not None:
                if self.useIT : 
                    if self.withHeatmap :
                        htmp = Image.fromarray(htmp)
                        tImage,htmp= self.transform(tImage,htmp)
                        
                        '''opencvImage = cv2.cvtColor(np.array(ti), cv2.COLOR_RGB2BGR)
                        htmp = np.array(ht)
                        
                        b_channel,g_channel,r_channel = opencvImage[:,:,0],opencvImage[:,:,1],opencvImage[:,:,2]
                        tImageCV = cv2.merge((b_channel, htmp,htmp, htmp))
                        cv2.imshow('t',tImageCV)
                        cv2.waitKey(0)
                        '''
                        heatmaps.append(htmp)
                    else :
                        tImage = self.transform(tImage,None)
                else : 
                    tImage = self.transforms['train'](tImage)
                #print(tImage,torch.max(tImage))
            aImgs.append(tImage)
        
        if self.mLBL : 
            
            theLabel = int(label[0])
            #print(theLabel)
            if self.reduce_label : 
                theLabel = np.argwhere(self.cIndex==theLabel)[0][0] + 1###shifted by one!!! 0 -999
                #print('cgd',theLabel,self.cIndex[theLabel-1],self.cIndex.shape,np.max(self.cIndex))
                #exit(0)
            #print(convertToOneHot(np.asarray([theLabel-1]), 1000).shape)
            if self.toOneHot : 
                gt = torch.LongTensor(convertToOneHot(np.asarray([theLabel-1]), 1000)).squeeze();
            else :
                if self.reduce_label : 
                    gt = torch.LongTensor([theLabel-1]);
                else : 
                    gt = torch.LongTensor([theLabel-1]);
            
            return aImgs[0],gt,heatmaps,ldmarks,limgs[0] 
        else : 
            
            img1 = aImgs[0]; img2 = aImgs[1]; 
            fold = torch.FloatTensor([int(label[0])]); 
            
            theLabel = int(label[1])
            
            if self.cGT : 
                if theLabel > 0 : 
                    theLabel *= 1 
                else : 
                    theLabel = -1
                    
            gt = torch.FloatTensor([theLabel]); 
            
            #print('f',fold,'gt',gt,label[0],label[1])
            #print(img1.size(),img2.size())
            
            if self.return_data : 
                return img1,img2,fold,gt,heatmaps,ldmarks,limgs[0],limgs[1]
            else : 
                return img1,img2,fold,gt,heatmaps,ldmarks

    
    def __len__(self):
        
        return len(self.l_imgs)
    
    
    def transform(self, img,htmp):
        
        '''img2 = np.array(img, dtype=np.uint8)
        cv2.imshow('tt',img2)
        cv2.waitKey(0)'''
        
        img = transforms.Resize(224)(img)
        if not self.isTest : 
            if not self.withHeatmap : 
                img = transforms.RandomResizedCrop(224,scale=(0.5, 1.0))(img)
                img = transforms.RandomHorizontalFlip()(img)
                img = transforms.RandomRotation(45)(img)
            else : 
                img,htmp = RandomResizedCrop_m(224,scale=(0.5, 1.0))(img,htmp)
                img,htmp = RandomHorizontalFlip_m()(img,htmp)
                img,htmp = RandomRotation_m(45)(img,htmp)
            
        imgTemp = np.array(img, dtype=np.uint8)
        #htmpTemp = np.array(htmp, dtype=np.uint8)
        '''cv2.imshow('tt',img)
        cv2.waitKey(0)'''
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = imgTemp.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        if self.withHeatmap : 
            return img,transforms.ToTensor()(htmp).float()
        else:  
            return img
    
    
    
    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl


class RandomResizedCrop_m(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img,htmp):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation),F.resized_crop(htmp, i, j, h, w, self.size, self.interpolation)




class RandomHorizontalFlip_m(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,htmp):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img),F.hflip(htmp)
        return img,htmp

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
    
    
class RandomRotation_m(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img,htmp):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center),F.rotate(htmp, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
