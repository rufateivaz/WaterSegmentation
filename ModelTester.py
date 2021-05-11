from Utils import saveImage, calculateF1, calculateMetrics, transform_to
from tqdm import tqdm
import numpy as np
import tensorflow as tf

DIR = '/data/'

# tests the performance for current batch.
@tf.function
def test_on_batch(x, y, seg_model):
    y_pred = seg_model.model(x, training=False)
    y_pred = tf.image.resize(y_pred, (1080, 1920), method=tf.image.ResizeMethod.BICUBIC)
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    seg_model.update_states(y, y_pred)

# tests the performance for current epoch.
def test(test_data, seg_model):
    for batch, (x, y) in enumerate(test_data):
        test_on_batch(x, y, seg_model) 
    # Get the test metrics for the current epoch
    test_seg_loss = seg_model.seg_loss.result().numpy()
    test_sensitivity, test_specificity, test_precision, test_accuracy, test_f1_score = calculateMetrics(seg_model.tp, seg_model.tn, seg_model.fp, seg_model.fn)   
        
    print ("- test_seg_loss: %.4f" % test_seg_loss, "- test_sensitivity: %.4f" % test_sensitivity,
    '- test_specificity: %.4f' % test_specificity, '- test_precision: %.4f' % test_precision,
    '- test_accuracy: %.4f' % test_accuracy, '- test_f1_score: %.4f' % test_f1_score)

    seg_model.reset_states()


def upscaler(prediction):
    prediction = tf.convert_to_tensor(prediction)
    prediction = tf.image.resize(prediction, (1080, 1920), method=tf.image.ResizeMethod.BICUBIC)
    prediction = tf.clip_by_value(prediction, 0, 1)
    return prediction

def calculateConfMatrixPerState(seg_model, state, set):
    for i in tqdm(range(len(set))):
        X = np.expand_dims(set[i][0], axis=0)
        y = np.expand_dims(set[i][1], axis=0)
        prediction = seg_model.model.predict(X)
        prediction = upscaler(prediction)
        seg_model.tp.update_state(y,  prediction)
        seg_model.fp.update_state(y,  prediction)
        seg_model.fn.update_state(y,  prediction)

    print ("F1 score of " +  state + " state: ", calculateF1(seg_model.tp, seg_model.fn, seg_model.fp))
    seg_model.reset_states()


def calculateF1scorePerState(seg_model, test_set, names):
    
    merge = zip(test_set[0], test_set[1], names[0])
    open_set = []
    dock_set = []
    channel_set = [] 
    for x in merge:
        if "open" in x[2]:
            open_set += [(x[0], x[1])]
        elif "dock" in x[2]:
            dock_set += [(x[0], x[1])]
        else :
            channel_set += [(x[0], x[1])]

    calculateConfMatrixPerState(seg_model, 'open', open_set)
    calculateConfMatrixPerState(seg_model, 'dock', dock_set)
    calculateConfMatrixPerState(seg_model, 'channel', channel_set)


def findImageWithMinimumF1score(seg_model, test_set):
    min = (1, -1)
    max = (0, -1)
    for i in tqdm(range(len(test_set[0]))):
        X = np.expand_dims(test_set[0][i], axis=0)
        y = np.expand_dims(test_set[1][i], axis=0)
        prediction = seg_model.model.predict(X)
        prediction = upscaler(prediction)
        seg_model.update_states(y, prediction) 
        f1 = calculateF1(seg_model.tp, seg_model.fn, seg_model.fp)
        if (f1 < min[0]):
            min = (f1, i)
        if (f1 > max[0]):
            max = (f1, i)
        seg_model.reset_states()
    return min, max



def labelRGB(x, y, prediction, imageName=None, architecture=0):
    # this part takes an image, its prediction through the network and ground truth segmentation mask
    # based on true negative and true positive, the pixels are labeled.

    # upscaling the prediction to the original resolution.
    prediction = upscaler(prediction)

    # converto from tensor to numpy.
    prediction = prediction.numpy()
    
    # reducing dimension
    prediction = np.squeeze(prediction,axis=0)
    
    # change mean-centerd to [0,1] range and change bgr to rgb if VGG-16 encoder or 
    x = transform_to(x, architecture)

    # upscaling the rgb image to the original resolution.
    x = upscaler(x)
    x = x.numpy()

    # Coloring the pixels based on prediction.
    x = np.where(prediction < 0.5, x + [0,0,1], x)
    x = np.where(prediction >= 0.5, x + [1,0,0], x)

    saveImage(x, imageName)


def testNetwork(test_data, test_set, seg_model, names, architecture):

    # Test the network with the given test set
    test(test_data, seg_model)

    # Separately calculate F1 score per state (open, dock, channel)
    calculateF1scorePerState(seg_model, test_set, names)

    # Showing best and worst predicted images.
    min, max = findImageWithMinimumF1score(seg_model, test_set)

    # Showing the best and the worst scores among all test set.
    test_img_min = np.copy(test_set[0][min[1]])
    test_lbl_min = np.copy(test_set[1][min[1]])
    test_img_max = np.copy(test_set[0][max[1]])
    test_lbl_max = np.copy(test_set[1][max[1]])

    
    # The worst and the best test image results are saved but before that, 
    # each of their pixels are colored (red: water, blue: non-water)     
    print ("Min F1 score is: " + str(min[0]))
    image_ = np.expand_dims(test_img_min, axis=0)
    prediction = seg_model.model.predict(image_)
    labelRGB(test_img_min, test_lbl_min, prediction, DIR + 'Best_Worst_Outputs/min.png', architecture)

    print ("Max F1 score is: " + str(max[0]))
    image_ = np.expand_dims(test_img_max, axis=0)
    prediction = seg_model.model.predict(image_)
    labelRGB(test_img_max, test_lbl_max, prediction, DIR + 'Best_Worst_Outputs/max.png', architecture)




