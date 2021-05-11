import math
import glob
import numpy as np
from tqdm import tqdm
import cv2
import tensorflow as tf
from Utils import custom_preprocessing, saveImage


DIR = '/data/'


# shuffles x and y arrays with the same indices.
def shuffler(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y

# Custom data generator for each batch. 
def customGenerator(images, labels, batch_size=16):
    length = len(images)
    if labels is None:
        np.random.shuffle(images)
    else:
        images, labels = shuffler(images, labels)
    while True:
        start = 0
        end = batch_size
        while True:
            if end < length:
                if labels is None:
                    x = np.copy(images[start:end])
                    yield x
                else :
                    x = np.copy(images[start:end])
                    y = np.copy(labels[start:end])
                    yield x, y
                start += batch_size
                end += batch_size

            else :
                if labels is None:
                    x1 = np.copy(images[start : length])
                    np.random.shuffle(images)
                    x2 = np.copy(images[0:(end - length)])
                    x = np.concatenate((x1, x2), axis=0)
                    yield x
                else :
                    x1 = np.copy(images[start : length])
                    y1 = np.copy(labels[start : length])
                    
                    # re-shuffling
                    images, labels = shuffler(images, labels)

                    x2 = np.copy(images[0:(end - length)])
                    y2 = np.copy(labels[0:(end - length)])
                    x = np.concatenate((x1, x2), axis=0)
                    y = np.concatenate((y1, y2), axis=0)
                    yield x, y
                start = end - length
                end = start + batch_size


# When supervised learning technique is used, the functions of this class is applied as preprocessing.
class SupervisedLearning():
    def __init__(self, directory, shape, batch_size, split_type, architecture):
        self.directory = directory 
        self.shape = shape
        self.batch_size = batch_size
        self.split_type = split_type
        self.architecture = architecture

    def generateImages(self):
        # each takes the corresponding image names 
        channel = sorted(glob.glob(self.directory + "/channel/" + "*.jpg")) 
        dock = sorted(glob.glob(self.directory + "/dock/" + "*.jpg"))
        op = sorted(glob.glob(self.directory + "/open/" + "*.jpg"))
        
        # each takes the corresponding labele names
        channel_mask = sorted(glob.glob(self.directory + "/channel_mask/" + "*.png")) 
        dock_mask = sorted(glob.glob(self.directory + "/dock_mask/" + "*.png"))
        op_mask = sorted(glob.glob(self.directory + "/open_mask/" + "*.png"))


        # Based on the given split, the left side of symbol '/' is divided between training and validation sets as 80/20, while the right side stands for test set.
        if self.split_type=='c/od': # channel / open + dock
            train_image_names = channel[:160]
            train_label_names = channel_mask[:160]
            val_image_names = channel[-40:]
            val_label_names = channel_mask[-40:]
            test_image_names = op + dock
            test_label_names = op_mask + dock_mask

        elif self.split_type=='d/co': # dock / channel + open
            train_image_names = dock[:160]
            train_label_names = dock_mask[:160]
            val_image_names = dock[-40:]
            val_label_names = dock_mask[-40:]
            test_image_names = channel + op
            test_label_names = channel_mask + op_mask

        elif self.split_type=='o/cd': # open / channel + dock
            train_image_names = op[:160]
            train_label_names = op_mask[:160]
            val_image_names = op[-40:]
            val_label_names = op_mask[-40:]
            test_image_names = channel + dock
            test_label_names = channel_mask + dock_mask

        elif self.split_type=='cd/o': # channel + dock / open
            train_image_names = channel[:160] + dock[:160]
            train_label_names = channel_mask[:160] + dock_mask[:160]
            val_image_names = channel[-40:] + dock[-40:]
            val_label_names = channel_mask[-40:] + dock_mask[-40:]
            test_image_names = op
            test_label_names = op_mask

        elif self.split_type=='co/d': # channel + open / dock
            train_image_names = channel[:160] + op[:160]
            train_label_names = channel_mask[:160] + op_mask[:160]
            val_image_names = channel[-40:] + op[-40:]
            val_label_names = channel_mask[-40:] + op_mask[-40:]
            test_image_names = dock
            test_label_names = dock_mask

        elif self.split_type=='do/c': # dock + open / channel
            train_image_names = dock[:160] + op[:160]
            train_label_names = dock_mask[:160] + op_mask[:160]
            val_image_names = dock[-40:] + op[-40:]
            val_label_names = dock_mask[-40:] + op_mask[-40:]
            test_image_names = channel
            test_label_names = channel_mask

        elif self.split_type=='a/c': # all / channel
            train_image_names = channel[:80] + op[:80] + dock[:80]
            train_label_names = channel_mask[:80] + op_mask[:80] + dock_mask[:80]
            val_image_names = channel[80:100] + op[80:100] + dock[80:100]
            val_label_names = channel_mask[80:100] + op_mask[80:100] + dock_mask[80:100]
            test_image_names = channel[-100:]
            test_label_names = channel_mask[-100:]

        elif self.split_type=='a/d': # all / dock
            train_image_names = channel[:80] + op[:80] + dock[:80]
            train_label_names = channel_mask[:80] + op_mask[:80] + dock_mask[:80]
            val_image_names = channel[80:100] + op[80:100] + dock[80:100]
            val_label_names = channel_mask[80:100] + op_mask[80:100] + dock_mask[80:100]
            test_image_names = dock[-100:]
            test_label_names = dock_mask[-100:]

        elif self.split_type=='a/o': # all / open
            train_image_names = channel[:80] + op[:80] + dock[:80]
            train_label_names = channel_mask[:80] + op_mask[:80] + dock_mask[:80]
            val_image_names = channel[80:100] + op[80:100] + dock[80:100]
            val_label_names = channel_mask[80:100] + op_mask[80:100] + dock_mask[80:100]
            test_image_names = op[-100:]
            test_label_names = op_mask[-100:]

        else:  # all / all
            train_image_names = channel[:80] + op[:80] + dock[:80]
            train_label_names = channel_mask[:80] + op_mask[:80] + dock_mask[:80]
            val_image_names = channel[80:100] + op[80:100] + dock[80:100]
            val_label_names = channel_mask[80:100] + op_mask[80:100] + dock_mask[80:100]
            test_image_names = channel[-100:] + dock[-100:] + op[-100:]
            test_label_names = channel_mask[-100:] + dock_mask[-100:] + op_mask[-100:]

        # the number of training images, validation images and test images.
        numberOfTrainImages = len(train_image_names)
        numberOfValImages = len(val_image_names)
        numberOfTestImages = len(test_image_names)

        # how many steps in one epoch is defined
        steps_per_epoch = math.ceil(numberOfTrainImages/self.batch_size)

        print("\n Loading Train images ...")
        train_images=self.loadImages(train_image_names, train_label_names, numberOfTrainImages, downscale=True)
        print("\n Loading Validation images ...")
        val_images=self.loadImages(val_image_names, val_label_names, numberOfValImages, downscale=False)
        print("\n Loading Test images ...")
        test_images=self.loadImages(test_image_names, test_label_names, numberOfTestImages, downscale=False)

        # prepare the train data, which is produced batchwise when next() function is called 
        train_data = customGenerator(train_images[0], train_images[1], self.batch_size)

        return train_data, val_images, test_images, steps_per_epoch, (test_image_names, test_label_names)

    # based on given image and corresponding label names, images and labeles are loaded to the images and labels as numpy arrays. 
    def loadImages(self, image_names, label_names, numberOfImages, downscale):
        # Downscale images that are used to train the network, otherwise leave them.
        images = np.zeros(shape=(numberOfImages, self.shape[0], self.shape[1], 3), dtype=np.float32)
        if downscale :
            labels = np.zeros(shape=(numberOfImages, self.shape[0], self.shape[1], 1), dtype=np.float32)
        else :
            labels = np.zeros(shape=(numberOfImages, 1080, 1920, 1), dtype=np.float32)

        # load images
        for i in tqdm(range(numberOfImages)):
            # read image, down-scale with bicubic interpolation, change to float32, apply preprocessing based on chosen architecture
            img = cv2.imread(image_names[i])
            img = cv2.resize(img, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
            img = np.float32(img)
            img = custom_preprocessing(img, architecture=self.architecture)
            images[i] = img

            # read label, down-scale if it is used for training, or leave it. 
            # expand dims since the last channel is not taken when using imread with grayscale.
            # change to float32 and normalize to [0,1] 
            lbl = cv2.imread(label_names[i], cv2.IMREAD_GRAYSCALE)
            if downscale:
                lbl = cv2.resize(lbl, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
            lbl = np.expand_dims(lbl, axis=2)
            lbl = np.float32(lbl)
            labels[i] = np.round(lbl / 255)
        
        return [images, labels]



# When domain adaptation technique is used, the functions of this class is applied as preprocessing.
class DomainAdaptation():
    def __init__(self, source_dir, target_dir, shape, batch_size, architecture):
        self.source_dir = source_dir
        self.target_dir = target_dir 
        self.shape = shape
        self.batch_size = batch_size
        self.architecture = architecture


    def generateImages(self):

        # Target train, validation and test sets are prepared
        channel = sorted(glob.glob(self.target_dir + "/channel/" + "*.jpg")) 
        dock = sorted(glob.glob(self.target_dir + "/dock/" + "*.jpg"))
        op = sorted(glob.glob(self.target_dir + "/open/" + "*.jpg"))
            
        channel_mask = sorted(glob.glob(self.target_dir + "/channel_mask/" + "*.png")) 
        dock_mask = sorted(glob.glob(self.target_dir + "/dock_mask/" + "*.png"))
        op_mask = sorted(glob.glob(self.target_dir + "/open_mask/" + "*.png"))


        # prepare target domain
        train_split = 80
        val_split = 100-train_split
        target_train_image_names = channel[:train_split] + op[:train_split] + dock[:train_split]
        val_image_names = channel[train_split:train_split+val_split] + op[train_split:train_split+val_split] + dock[train_split:train_split+val_split]
        val_label_names = channel_mask[train_split:train_split+val_split] + op_mask[train_split:train_split+val_split] + dock_mask[train_split:train_split+val_split]
        test_image_names = channel[-100:] + dock[-100:] + op[-100:]
        test_label_names = channel_mask[-100:] + dock_mask[-100:] + op_mask[-100:]

        # source train set is prepared
        if self.source_dir == 1:
            # load source images : only MaSTr1325
            directory = '/data/MaSTr1325'
            source_image_names = sorted(glob.glob(directory + "/images/" + "*.jpg"))
            source_label_names = sorted(glob.glob(directory + "/labels/" + "*.png"))
            numberOfSourceImages = len(source_image_names)
            print("\n Loading Source images ...")
            source_images=self.loadImages(source_image_names, source_label_names, numberOfSourceImages, downscale=True, convertToBinary=True)

        elif self.source_dir == 2:
            # load source images : only WaterDataset2
            directory = '/data/Rufat/WaterDataset2'
            source_image_names = sorted(glob.glob(directory + "/images/river_segs/" + "*.jpg")) + sorted(glob.glob(directory + "/images/ADE20K/" + "*.png"))     
            source_label_names = sorted(glob.glob(directory + "/labels/river_segs/" + "*.png")) + sorted(glob.glob(directory + "/labels/ADE20K/" + "*.png")) 
            numberOfSourceImages = len(source_image_names)
            print("\n Loading Source images ...")
            source_images=self.loadImages(source_image_names, source_label_names, numberOfSourceImages, downscale=True)

        else :
            # load source images : both MaSTr1325 and WaterDataset2
            directory = '/data/MaSTr1325'
            source_image_names = sorted(glob.glob(directory + "/images/" + "*.jpg"))
            source_label_names = sorted(glob.glob(directory + "/labels/" + "*.png"))
            numberOfSource1Images = len(source_image_names)
            print("\n Loading Source images (MaSTr1325)")
            source_images1=self.loadImages(source_image_names, source_label_names, numberOfSource1Images, downscale=True, convertToBinary=True)

            directory = '/data/Rufat/WaterDataset2'
            source_image_names =   sorted(glob.glob(directory + "/images/river_segs/" + "*.jpg")) + sorted(glob.glob(directory + "/images/ADE20K/" + "*.png"))    
            source_label_names =   sorted(glob.glob(directory + "/labels/river_segs/" + "*.png")) + sorted(glob.glob(directory + "/labels/ADE20K/" + "*.png"))
            numberOfSource2Images = len(source_image_names)
            print("\n Loading Source images (WaterDataset)")
            source_images2=self.loadImages(source_image_names, source_label_names, numberOfSource2Images, downscale=True)

            source_images = []
            source_images.append(np.concatenate((source_images1[0], source_images2[0]), axis=0))
            source_images.append(np.concatenate((source_images1[1], source_images2[1]), axis=0))
            numberOfSourceImages = numberOfSource1Images + numberOfSource2Images

        print ("source_images: ", source_images[0].shape, source_images[1].shape, np.min(source_images[0]), np.max(source_images[0]), np.min(source_images[1]), np.max(source_images[1]))

        
        saveImage(source_images[0][0], DIR + 'Best_Worst_Outputs/' + 's1i.png')
        saveImage(source_images[1][0], DIR + 'Best_Worst_Outputs/' + 's1m.png')

        """
        cv2.imwrite("Best_Worst_Outputs/i1.png", source_images[0][1325] * 255)
        cv2.imwrite("Best_Worst_Outputs/l1.png", source_images[1][1325] * 255)
        cv2.imwrite("Best_Worst_Outputs/i2.png", source_images[0][1325 + 1888] * 255)
        cv2.imwrite("Best_Worst_Outputs/l2.png", source_images[1][1325 + 1888] * 255)
        cv2.imwrite("Best_Worst_Outputs/i3.png", source_images[0][1325 + 1888 + 299] * 255)
        cv2.imwrite("Best_Worst_Outputs/l3.png", source_images[1][1325 + 1888 + 299] * 255)
        """

        numberOfTargetTrainImages = len(target_train_image_names)
        numberOfValidationImages = len(val_image_names)
        numberOfTestImages = len(test_image_names)
        
        # defines steps per epoch
        steps_per_epoch = math.ceil(len(source_images[0]) / self.batch_size)

        print("\n Loading Target train images ...")
        target_train_images = self.loadImages(target_train_image_names, None,  numberOfTargetTrainImages, downscale=True)
        print("\n Loading Validation images ...")
        val_images = self.loadImages(val_image_names, val_label_names,  numberOfValidationImages, downscale=False)
        print("\n Loading Test images ...")
        test_images = self.loadImages(test_image_names, test_label_names, numberOfTestImages, downscale=False)

        # prepare the source train data and target train data, which are produced batchwise when next() function is called.
        source_train = customGenerator(source_images[0], source_images[1], self.batch_size)
        target_train = customGenerator(target_train_images, None, self.batch_size)

        return source_train, target_train, val_images, test_images, steps_per_epoch, (test_image_names, test_label_names) 

    # based on given image and corresponding labele names, images and labeles are loaded to the images and labels numpy arrays. 
    def loadImages(self, image_names, label_names, numberOfImages, downscale, convertToBinary=False):
        images = np.zeros(shape=(numberOfImages, self.shape[0], self.shape[1], 3), dtype=np.float32)
        if downscale :
            labels = np.zeros(shape=(numberOfImages, self.shape[0], self.shape[1], 1), dtype=np.float32)
        else :
            labels = np.zeros(shape=(numberOfImages, 1080, 1920, 1), dtype=np.float32)
        for i in tqdm(range(numberOfImages)):
            # read image, down-scale with bicubic interpolation, change to float32, apply pre-processing based on chosen architecture.
            img = cv2.imread(image_names[i])
            img = cv2.resize(img, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
            img = np.float32(img)
            img = custom_preprocessing(img, architecture=self.architecture)
            images[i] = img

            # source domain
            if label_names is not None:
                lbl = cv2.imread(label_names[i], cv2.IMREAD_GRAYSCALE)
                # in MaSTr1325, there are four classes. We convert them to 0/1 class. 
                if convertToBinary:
                    lbl[lbl != 1] = 0
                    lbl[lbl == 1] = 255
                if downscale:
                    lbl = cv2.resize(lbl, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
                lbl = np.expand_dims(lbl, axis=2)
                lbl = np.float32(lbl)
                labels[i] = np.round(lbl / 255)
        if label_names is not None: 
            return [images, labels]
        else :
            return images
