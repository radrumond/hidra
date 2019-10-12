import numpy as np
import os
import cv2
import pickle


    
class MiniImgNet_Gen: 
    
    def __init__(self,path="/tmp/data/miniimagenet",data_path=None):  
        
        if data_path is None:
            self.path = path
            self.train_paths =  ["train/"+x for x in os.listdir(path+"/train")]
            self.test_paths =  ["test/"+x for x in os.listdir(path+"/test")]
            self.val_paths =  ["val/"+x for x in os.listdir(path+"/val")]

        self.data_path  = data_path
        self.meta_train = None
        self.meta_test  = None
        self.meta_val   = None
    
    
    def sample_Task(self,mb_size, min_class,max_class,train_size,test_size,training="train",shuffle=True):

        print('Loading MiniImagenet data...')
        if training == "train": 
            if self.meta_train is None:  
                meta_data = []
                for idx,im_class in enumerate(self.train_paths):
                    meta_data.append(np.array(loadImgDir(self.path+"/"+im_class,[84,84],rgb=True)))
                self.meta_train = meta_data
            else:
                meta_data = self.meta_train
                

        elif training == "val":   
            if self.meta_val is None:  
                meta_data = []
                for idx,im_class in enumerate(self.val_paths):
    #                 print(idx)
                    meta_data.append(np.array(loadImgDir(self.path+"/"+im_class,[84,84],rgb=True)))
                self.meta_val = meta_data
            else:
                meta_data = self.meta_val
                

        elif training == "test": 
            if self.meta_test is None: 
                meta_data = []
                for idx,im_class in enumerate(self.test_paths):
    #                 print(idx)
                    meta_data.append(np.array(loadImgDir(self.path+"/"+im_class,[84,84],rgb=True)))
                self.meta_test = meta_data
            else:
                meta_data = self.meta_test
                
        else:
            raise ValueError("Training needs to be train, val or test")
        print(f'Finished loading MiniImagenet data: {np.array(meta_data).shape}')       
                
        if min_class < 2:
            raise ValueError("Minimum number of classes must be >=2")

            
        
        while True:

            meta_train_x = []
            meta_train_y = []
            meta_test_x  = []
            meta_test_y  = []
            
            # sample fixed number classes for a meta batch
            nr_classes = np.random.randint(min_class,max_class)
            
            
            for mb in range(mb_size):

                # select which classes in the meta batch
                classes = np.random.choice(range(len(meta_data)),nr_classes,replace=False)
                train_x = []
                train_y = []
                test_x  = []
                test_y  = []
            
                for label_nr,cl in enumerate(classes):

                    images = np.random.choice(len(meta_data[cl]),train_size+test_size,False)
                    train_imgs = images[:train_size]
                    test_imgs  = images[train_size:]
            
                    train_x.append(meta_data[cl][train_imgs])
                    test_x.append(meta_data[cl][test_imgs])

                    train_y.append(np.ones(train_size)*label_nr)
                    test_y.append(np.ones(test_size)*label_nr)
                    
                    
                train_x = np.array(train_x)
                train_y = np.eye(len(classes))[np.reshape(np.array(train_y),-1).astype(int)]
                test_x = np.array(test_x)
                test_y = np.eye(len(classes))[np.reshape(np.array(test_y),-1).astype(int)]

                train_x = np.reshape(train_x,[-1,84,84,3])
                test_x = np.reshape(test_x,[-1,84,84,3])
                
                if shuffle:
                    train_x,train_y = unison_shuffled_copies(train_x,train_y)
                    test_x,test_y = unison_shuffled_copies(test_x,test_y)
                    
                meta_train_x.append(train_x)
                meta_train_y.append(train_y)
                meta_test_x.append(test_x)
                meta_test_y.append(test_y)  
            # print('YIEEEEEEELDING')
            yield meta_train_x,meta_train_y,meta_test_x,meta_test_y



# Initiates the Omniglot dataset and splits into meta train and meta task
class OmniChar_Gen:
    
    def __init__(self,path="/tmp/data/omniglot",data_path=None,test_idx=None):

        self.path = path
        self.tasks = ["/images_background/"+x for x in os.listdir(path+"/images_background")]+["/images_evaluation/"+x for x in os.listdir(path+"/images_evaluation")]
        
 
        self.lens = {}
        for task in self.tasks:
            self.lens[task] = len(os.listdir(self.path+task))
            
        self.meta_data = []
        print("Loading Omniglot data")
        for idx,task in enumerate(range(len(self.tasks))):
            if idx%10==0:
                print(f"Loading tasks {idx}/{len(self.tasks)}")
            data = []
            for char in os.listdir(self.path+self.tasks[task]):
                c = []

                for img in os.listdir(self.path+self.tasks[task]+"/"+char):
                    c.append(readImg(self.path+self.tasks[task]+"/"+char+"/"+img))

                data.append(c)
    
            self.meta_data.append(data)
        self.meta_data = np.concatenate(self.meta_data)

        print("Finished loading data")
        if test_idx==None:
            self.train_idx = list(range(len(self.meta_data)))
            np.random.shuffle(self.train_idx)
            self.test_idx  = self.train_idx[1200:]
            self.train_idx = self.train_idx[:1200]
            print("Test_idx:",self.test_idx)
        else:
            self.test_idx  = test_idx
            self.train_idx = list(set(list(range(len(self.meta_data)))) - set(self.test_idx))
        
    # Builds a generator that samples meta batches from meta training/test data
    def sample_Task(self,mb_size, min_class,max_class,train_size,test_size,training="train",shuffle=True):
        
        if training == "train":
            idx = self.train_idx
        elif training == "test":
            idx = self.test_idx
        else:
        	raise ValueError("Omniglot only supports train and test for training param")
        
        if min_class < 2:
            raise ValueError("Minimum number of classes must be >=2")
        ## We can remove this later and make it dynamic

        while True:
            
            image_idx = idx.copy()
            np.random.shuffle(image_idx)
                    
            meta_train_x = []
            meta_train_y = []
            meta_test_x = []
            meta_test_y = []
            
            # Roll number of classes in the mb
            nr_classes = np.random.randint(min_class,max_class)

            for task in range(mb_size):
                
                train_x = []
                train_y = []
                test_x = []
                test_y = []
                # Sample the characters for the task
                chars = np.random.choice(image_idx,nr_classes,False)

                # Sample the shots for each character
                for label_nr,char in enumerate(chars):
                    images = np.random.choice(range(20),train_size+test_size,False)
                    train_imgs = images[:train_size]
                    test_imgs  = images[train_size:]
                    
                    train_x.append(self.meta_data[char][train_imgs])
                    test_x.append(self.meta_data[char][test_imgs])

                    train_y.append(np.ones(train_size)*label_nr)
                    test_y.append(np.ones(test_size)*label_nr)
                    
                train_x = np.array(train_x)
                train_y = np.eye(len(chars))[np.reshape(np.array(train_y),-1).astype(int)]
                test_x = np.array(test_x)
                test_y = np.eye(len(chars))[np.reshape(np.array(test_y),-1).astype(int)]

                train_x = np.reshape(train_x,[-1,28,28,1])
                test_x = np.reshape(test_x,[-1,28,28,1])
                if shuffle:
                    train_x,train_y = unison_shuffled_copies(train_x,train_y)
                    test_x,test_y = unison_shuffled_copies(test_x,test_y)
                
                meta_train_x.append(train_x)
                meta_train_y.append(train_y)
                meta_test_x.append(test_x)
                meta_test_y.append(test_y)
                
            yield meta_train_x,meta_train_y,meta_test_x,meta_test_y
    
def getOrder(minClass,maxClass,mb_size,number_chars=1200):
    # gives a list integers between minClass and maxClass that sum up to 1200, 
    lens = []
    sums = 0
    while sums<=number_chars-minClass*mb_size:
        maxV = int((number_chars-sums)/mb_size)+1
        
        n=np.random.randint(minClass,min(maxV,maxClass))
        
        lens += [n]*mb_size
        sums  = sums+(n*mb_size) 
    return lens
    
def readImg(path,size=[28,28],rgb=False):
    
    img = cv2.imread(path)
    img = cv2.resize(img,(size[0],size[1])).astype(float)
    if np.max(img)>1.0:
        img /= 255.
    
    if not rgb:
        return img[:,:,:1]
    else:      

        if len(img.shape)==3:
            if img.shape[-1]!=3:
                print('ASFASFASFAS')
                print(img.shape)
                print(path)
            return img
        else:
            return np.reshape([img,img,img],[size[0],size[1],3])


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

                
def loadImgDir(path,size,rgb):
    
    imgs = []
    
    for img in os.listdir(path):
        
        imgs.append(readImg(path+"/"+img,size,rgb))
    return imgs
    
        
    