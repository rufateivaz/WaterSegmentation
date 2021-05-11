import keras.backend as K

def _true_positives(y_true, y_pred):
    return K.sum(y_true * K.round(y_pred))

def _true_negatives(y_true, y_pred):
    yt = (y_true - 1) * -1
    yp = (y_pred - 1) * -1
    return _true_positives(yt, yp)

def _false_negatives(y_true, y_pred):
    return K.sum(K.clip((y_true - K.round(y_pred)), 0, 1))

def _false_positives(y_true, y_pred):
    return _false_negatives(K.round(y_pred), y_true)

def calculateMetrics(confs):
    val_sensitivity = confs[0] / (confs[0] + confs[3] + K.epsilon())
    val_specificity = confs[1] / (confs[1] + confs[2] + K.epsilon())
    val_precision   = confs[0] / (confs[0] + confs[2] + K.epsilon())
    val_accuracy    = (confs[0] + confs[1]) / (confs[0] + confs[1] + confs[2] + confs[3] + K.epsilon())
    val_f1_score    = 2 * val_sensitivity * val_precision / (val_sensitivity + val_precision + K.epsilon())
    return val_sensitivity, val_specificity, val_precision, val_accuracy, val_f1_score


def getConfs(yt, yp):
    return [_true_positives(yt, yp), _true_negatives(yt, yp), 
            _false_positives(yt, yp), _false_negatives(yt, yp)]
