from albumentations.augmentations.functional import crop
from albumentations.augmentations.transforms import RandomCrop, RandomResizedCrop
from albumentations.core.composition import OneOf
from tensorflow.keras.utils import Progbar
from Utils import calculateMetrics, transform_to, transform_back
import albumentations as album
import tensorflow as tf
import numpy as np

class SupervisedLearner():
    def __init__(self, patience, save_to, seg_model, architecture):
        self.patience = patience
        self.save_to = save_to
        self.seg_model = seg_model
        self.saved_loss = float("inf")
        self.architecture = architecture


# Used albumentation for augmentation.
    def transformation(self, x, y):
        if x is None or y is None:
            raise ValueError("\nThe problem is in transformation function in DomainAdaptation.py; make sure that you write the values (xs,ys,xt) correctly.\n")
        
        x = transform_to(x, self.architecture)

        color_aug = album.Compose([        
            album.RGBShift(50, 50, 50, p=0.5)
        ], p=0.25)

        geo_aug = album.Compose([
            album.OneOf([
                album.Flip(p=0.5),
                album.ElasticTransform(interpolation=2, border_mode=4, p=0.5)
            ])
        ], p=0.25)

        for i in range(len(x)):
            x[i] = color_aug(image=x[i])['image']
            geo = geo_aug(image=x[i], mask=y[i])
            x[i], y[i] = geo['image'], geo['mask']
        x = transform_back(x, self.architecture)
        return x, np.clip(y, 0, 1)

    
    # EarlyStopper simply stops the training if there is no improvement in the performance after defined patience value.
    def EarlyStopper(self, val_seg_loss, patience):
        if (self.saved_loss <= val_seg_loss): # Not improved
            print("-------------------------------------------------------------------------------------------------------")
            print("The model did not improve from %.4f" % self.saved_loss)
            print("-------------------------------------------------------------------------------------------------------")
            patience += 1
            if (patience == self.patience):
                return patience, True
            else :
                return patience, False
        else : # improved
            print("-------------------------------------------------------------------------------------------------------")
            print("Improved from %.4f" % self.saved_loss + " to %.4f" % val_seg_loss + ".Current model is saved to " + self.save_to)
            print("-------------------------------------------------------------------------------------------------------")
            self.saved_loss = val_seg_loss
            self.seg_model.model.save(self.save_to)
            patience = 0
            return patience, False
    
    
    # validates the performance for current batch.
    @tf.function
    def validate_on_batch(self, X, y):
        y_pred = self.seg_model.model(X, training=False)
        y_pred = tf.image.resize(y_pred, (1080, 1920), method=tf.image.ResizeMethod.BICUBIC)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        self.seg_model.update_states(y, y_pred)

    # validates the performance for current epoch.
    def validate(self, val_data):
        for batch, (X, y) in enumerate(val_data):
            self.validate_on_batch(X, y) 
        
        # Get the validation metrics for the current epoch
        val_seg_loss = self.seg_model.seg_loss.result().numpy()
        val_sensitivity, val_specificity, val_precision, val_accuracy, val_f1_score = calculateMetrics(self.seg_model.tp, self.seg_model.tn, self.seg_model.fp, self.seg_model.fn)   
        
        # Add validation metrics to history for plotting graph after training the model is over.
        self.seg_model.val_history.append((val_seg_loss, val_sensitivity, val_specificity, val_precision, val_accuracy, val_f1_score))

        print ("- val_seg_loss: %.4f" % val_seg_loss, "- val_sensitivity: %.4f" % val_sensitivity,
        '- val_specificity: %.4f' % val_specificity, '- val_precision: %.4f' % val_precision,
        '- val_accuracy: %.4f' % val_accuracy, '- val_f1_score: %.4f' % val_f1_score)
        return val_seg_loss
    
    # trains the current batch
    @tf.function
    def train_on_batch(self, X, y):
        with tf.GradientTape() as tape:
            ps =  self.seg_model.model(X, training=True)
            seg_loss = self.seg_model.loss(y, ps)
        trainable_variables = self.seg_model.model.trainable_variables
        grads = tape.gradient(seg_loss, trainable_variables)
        self.seg_model.optimizer.apply_gradients(zip(grads, trainable_variables))
        self.seg_model.update_states(y, ps)
        return seg_loss

    # trains the current epoch
    def train(self, train_data, steps_per_epoch, progressBar):
        # Train the network in the current epoch
        for batch in range(steps_per_epoch):
            # Polynomial decay of the learning rate
            #self.seg_model.updateLearningRate(0.8)
            
            # get the current batch
            x, y = train_data.__next__()

            # apply transformation if needed
            #x, y = self.transformation(x, y)

            # update the weights based on the binary cross-entropy loss
            seg_loss = self.train_on_batch(x, y)
            
            # print train metric - loss
            values=[('loss', seg_loss)]
            progressBar.add(1, values=values)
            
        # Add train metrics to history for plotting graph later.
        train_loss = self.seg_model.seg_loss.result().numpy()
        sensitivity, specificity, precision, accuracy, f1_score = calculateMetrics(self.seg_model.tp, self.seg_model.tn, self.seg_model.fp, self.seg_model.fn)
        self.seg_model.train_history.append((train_loss, sensitivity, specificity, precision, accuracy, f1_score))


    def fit(self, train_data, val_data, epochs, steps_per_epoch):
        # metrics, used to show batch loss
        metrics_names=['loss']
        # Earlystopper counter
        patience = 0
        for epoch in range(0, epochs):
            # Progressbar showing printing each epoch
            print("\nepoch {}/{}".format(epoch+1,epochs))
            progressBar = Progbar(steps_per_epoch, stateful_metrics=metrics_names)
            
            # Training step
            self.train(train_data, steps_per_epoch, progressBar)
            # Reset metrics
            self.seg_model.reset_states()

            # validation step
            val_seg_loss = self.validate(val_data)
            # Reset metrics
            self.seg_model.reset_states()
            
            # EarlyStopping operations
            patience, stop_training = self.EarlyStopper(val_seg_loss, patience)
            
            # stop training if the conditions in Early Stopper holds
            if stop_training:
                break 
        # take the best model with less validation loss.
        self.seg_model.model = tf.keras.models.load_model(self.save_to, compile=False)