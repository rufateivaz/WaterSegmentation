import keras.backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

MEAN = [103.939, 116.779, 123.68]

# Plots graph for the loss or f1 score parameters obtained in training and validating phases.
def plotGraph(ylabel, xlabel, save_to, index, model, location):
    plt.plot([x[index] for x in model.train_history])
    plt.plot([x[index] for x in model.val_history])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Training', 'Validation'], loc=location)
    plt.savefig(save_to)
    #plt.show()
    plt.cla()

# saves the given image to the given name.
def saveImage(img, imageName):
    image.save_img(imageName, img)

# Below five different metrics are calculated based on given confusion matrix parameters: (tp, tn, fp, fn)
def calculateSensitivity(tp, fn):
    return tp.result().numpy() / (tp.result().numpy() + fn.result().numpy() + K.epsilon())

def calculateSpecificity(tn, fp):
    return tn.result().numpy() / (tn.result().numpy() + fp.result().numpy() + K.epsilon())

def calculatePrecision(tp, fp):
    return tp.result().numpy() / (tp.result().numpy() + fp.result().numpy() + K.epsilon())

def calculateAccuracy(tp, tn, fp, fn):
    return (tp.result().numpy() + tn.result().numpy()) / (tp.result().numpy() + tn.result().numpy() + fp.result().numpy() + fn.result().numpy() + K.epsilon())

def calculateF1(tp, fn, fp):
    se  = calculateSensitivity(tp, fn)
    pr  = calculatePrecision(tp, fp)
    return 2 * se * pr / (se + pr + K.epsilon())

def calculateMetrics(tp, tn, fp, fn):
    se = calculateSensitivity(tp, fn)
    sp = calculateSpecificity(tn, fp)
    prc = calculatePrecision(tp, fp)
    acc = calculateAccuracy(tp, tn, fp, fn)
    f1  = calculateF1(tp, fn, fp)
    return se, sp, prc, acc, f1


# This function is required for applying pre-processing while loading the images.
# Note that, for ResNet-50 and VGG-16 base models have been implemented by using keras.applications,
# where the pre-requisite is to change image colors to BGR and applying mean centered w.r.t ImageNet is important.
# The images are loaded with OpenCV's imread function that by default, loads images in BGR form.
# However, for UNet and lightweight (proposed in Tampere-WaterSeg), we use [0,1] range with RGB format.  
def custom_preprocessing(x, architecture=0):
    if architecture==2 or architecture==3:
        x[..., 0] -= MEAN[0]
        x[..., 1] -= MEAN[1]
        x[..., 2] -= MEAN[2]
        return x
    else: 
        x = x[...,::-1]
        x = x / 255
        return x

# Before applying augmentation, this function is required for shifting bgr values to rgb and 
# changing their range from mean centered w.r.t ImageNet, to [0,1]. (IF VGG-16 encoder or PSPNet is used)
def transform_to(x, architecture):
    # PSPNET and VGG-16 encoder
    if architecture == 2 or architecture==3:
        x[..., 0] += MEAN[0]
        x[..., 1] += MEAN[1]
        x[..., 2] += MEAN[2]
        x = x[...,::-1]
        x = x / 255
        return x
    # UNet and lightweight: Do nothing, since for those architecture, the range is set to [0,1] and colors are in RGB format.
    else : 
        return x

# After applying augmentation, this function is required for shifting rgb values to bgr and 
# changing their range from [0,1] back to mean centered w.r.t ImageNet. (IF VGG-16 encoder or PSPNet is used.)
def transform_back(x, architecture):
    # PSPNET and VGG-16 encoder
    if architecture == 2 or architecture==3:
        x = x * 255
        x = x[...,::-1]
        x[..., 0] -= MEAN[0]
        x[..., 1] -= MEAN[1]
        x[..., 2] -= MEAN[2]
        return x
    # UNet and lightweight: Do nothing, since for those architecture, the range is set to [0,1] and colors are in RGB format.
    else : 
        return x