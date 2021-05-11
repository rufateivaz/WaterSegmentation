from albumentations.augmentations.transforms import Flip, RandomRotate90
from tensorflow.keras.utils import Progbar
from Utils import calculateMetrics, transform_to, transform_back, saveImage
import albumentations as album
import tensorflow as tf
import numpy as np
import math

class DomainAdaptor():
    def __init__(self, patience, save_to, seg_model, disc_model, architecture):
        self.patience = patience
        self.save_to = save_to
        self.seg_model = seg_model
        self.disc_model = disc_model
        self.saved_loss = float("inf")
        self.adv_lambda = 0.001
        self.architecture = architecture
    
    # augmentation part for source and/or target domains.
    def transformation(self, xs = None, ys=None, xt=None, size=8):
        if xs is None or ys is None or xt is None:
            raise ValueError("\nThe problem is in transformation function in DomainAdaptation.py; make sure that you write the values (xs,ys,xt) correctly.\n")
        
        #transform_source = album.RandomResizedCrop(always_apply=False, p=1.0, height=256, width=256, scale=(0.07999999821186066, 1.0), ratio=(0.75, 1.4900000095367432), interpolation=2)
        #transform_target = album.RandomResizedCrop(always_apply=False, p=1.0, height=256, width=256, scale=(0.07999999821186066, 1.0), ratio=(0.75, 1.4900000095367432), interpolation=2)
        transform_source = album.RandomGridShuffle(grid=(4, 8), p=0.8)
        transform_target = album.RandomGridShuffle(grid=(4, 8), p=0.8)

        # since the pixel values are mean centered w.r.t ImageNet
        # we first change the pixel values to [0,1]
        # apply augmentations
        # then change the pixel values back to mean centered w.r.t ImageNet
        
        xs = transform_to(xs, self.architecture)
        xt = transform_to(xt, self.architecture)
        for i in range(len(xs)):
            # apply augmentation to source domain.
            transform = transform_source(image=xs[i], mask=ys[i])
            xs[i], ys[i]  = transform['image'], transform['mask']
            
            # apply augmentation to target domain.            
            transform = transform_target(image=xt[i])
            xt[i] = transform['image']

        xs = transform_back(xs, self.architecture)
        xt = transform_back(xt, self.architecture)

        return xs, np.clip(ys, 0, 1), xt            


        
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
    def val_on_batch(self, X, y):
        # Update mean of validation loss after validating current batch. Also, accumulate number of tps, tns, fps, fns members of confusion matrix.
        y_pred = self.seg_model.model(X, training=False)
        pred = self.disc_model.model(y_pred, training=False)
        self.disc_model.update_states(tf.ones_like(pred), tf.zeros_like(pred), pred)
        y_pred = tf.image.resize(y_pred, (1080, 1920), method=tf.image.ResizeMethod.BICUBIC)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        self.seg_model.update_states(y, y_pred)

    # validates the performance for current epoch.
    def validate(self, val_data):
        # validates the network in current epoch
        for batch, (X, y) in enumerate(val_data):
            self.val_on_batch(X, y)   
        
        # Get the validation metrics for the current epoch
        val_seg_loss = self.seg_model.seg_loss.result().numpy()
        val_adv_loss = self.disc_model.adv_loss.result().numpy()
        val_disc_loss = self.disc_model.disc_loss.result().numpy()
        val_sensitivity, val_specificity, val_precision, val_accuracy, val_f1_score = calculateMetrics(self.seg_model.tp, self.seg_model.tn, self.seg_model.fp, self.seg_model.fn)   
        
        # Add validation metrics to history for plotting graph after training the model is over.
        self.seg_model.val_history.append((val_seg_loss, val_adv_loss, val_disc_loss, val_sensitivity, val_specificity, val_precision, val_accuracy, val_f1_score))

        print ("- val_seg_loss: %.4f" % val_seg_loss, "- val_adv_loss: %.4f" % val_adv_loss, "- val_disc_loss: %.4f" % val_disc_loss, 
        "- val_sensitivity: %.4f" % val_sensitivity, '- val_specificity: %.4f' % val_specificity, '- val_precision: %.4f' % val_precision,
        '- val_accuracy: %.4f' % val_accuracy, '- val_f1_score: %.4f' % val_f1_score)
        return val_seg_loss
    

    @tf.function
    def train_on_batch_nouda(self, xs, ys):
        with tf.GradientTape() as tape:
            # get the segmentation loss for source domain
            ps =  self.seg_model.model(xs, training=True)
            seg_loss = self.seg_model.loss(ys, ps)
        # computer gradients
        seg_grads = tape.gradient(seg_loss, self.seg_model.model.trainable_variables)
        # update weights
        self.seg_model.optimizer.apply_gradients(zip(seg_grads, self.seg_model.model.trainable_variables))
        # accumulate segmentation, discriminator and adversarial losses for plotting the graphs of each of them after the training is over.
        self.seg_model.update_states(ys, ps)
        loss = seg_loss
        adv_loss = 0
        dloss = 0
        return loss, adv_loss, dloss

    @tf.function
    def train_on_batch_uda(self, xs, ys, xt):
        
        with tf.GradientTape() as tape_seg, tf.GradientTape() as tape_disc:
            # get the segmentation loss for source domain
            ps =  self.seg_model.model(xs, training=True)
            seg_loss = self.seg_model.loss(ys, ps)

            # get the adversarial loss acquired by target unlabeled data.
            pt =  self.seg_model.model(xt, training=True)
            dpt =  self.disc_model.model(pt, training=False)
            adv_loss = self.disc_model.loss(tf.ones_like(dpt), dpt)

            loss = seg_loss + adv_loss * self.adv_lambda 

            # train discriminator by coming output from the segmentation network, to which the input is given
            # either from source domain (loss is calculated by ones) or from target domain (loss is calculated by zeros) 
            # ones for source , zeros for target. Then, add them and take the mean of these losses.
            
            sp = self.seg_model.model(xs, training=False)
            dsp =  self.disc_model.model(sp, training=True)
            ds_loss = self.disc_model.loss(tf.ones_like(dsp), dsp) 

            tp = self.seg_model.model(xt, training=False)
            dtp =  self.disc_model.model(tp, training=True)
            dt_loss = self.disc_model.loss(tf.zeros_like(dtp), dtp) 

            dloss = (ds_loss + dt_loss) * 0.5

        seg_grads = tape_seg.gradient(loss, self.seg_model.model.trainable_variables)
        disc_grads = tape_disc.gradient(dloss, self.disc_model.model.trainable_variables)
        
        self.seg_model.optimizer.apply_gradients(zip(seg_grads, self.seg_model.model.trainable_variables))
        self.disc_model.optimizer.apply_gradients(zip(disc_grads, self.disc_model.model.trainable_variables))
        
        # accumulate segmentation, discriminator and adversarial losses for plotting the graphs of each of them after the training is over.
        self.seg_model.update_states(ys, ps)
        self.disc_model.adv_loss.update_state(tf.ones_like(dpt), dpt)
        self.disc_model.disc_loss.update_state(tf.ones_like(dsp), dsp)
        self.disc_model.disc_loss.update_state(tf.zeros_like(dtp), dtp)

        return seg_loss, adv_loss, dloss

    def train(self, source, target, steps_per_epoch, progressBar, epoch):
        # Train the network in the current epoch
        for batch in range(steps_per_epoch):
            xs, ys = source.__next__()
            xt = target.__next__()

            #xs, ys, xt = self.transformation(xs, ys, xt)

            #with adaptation
            seg_loss, adv_loss, disc_loss = self.train_on_batch_uda(xs, ys, xt)
            
            # without adaptation
            #seg_loss, adv_loss, disc_loss = self.train_on_batch_nouda(xs, ys)

            values=[('seg_loss', seg_loss), ('adv_loss', adv_loss), ('disc_loss', disc_loss)]
            progressBar.add(1, values=values)
           
        # Add train metrics to history for plotting graph later.
        train_seg_loss = self.seg_model.seg_loss.result().numpy()
        train_adv_loss = self.disc_model.adv_loss.result().numpy()
        train_disc_loss = self.disc_model.disc_loss.result().numpy()
        sensitivity, specificity, precision, accuracy, f1_score = calculateMetrics(self.seg_model.tp, self.seg_model.tn, self.seg_model.fp, self.seg_model.fn)
        self.seg_model.train_history.append((train_seg_loss, train_adv_loss, train_disc_loss, sensitivity, specificity, precision, accuracy, f1_score))

  

    def fit(self, source, target, val_data, epochs, steps_per_epoch):
        # Initialize maximum iterations for polynomial decay the learning rate of both networks: Segmentation and Discriminator.
        # metrics, used to show batch loss
        metrics_names=['seg_loss', 'adv_loss', 'disc_loss']
        # Earlystopper counter
        patience = 0
        for epoch in range(0, epochs):
            # Progressbar showing printing each epoch
            print("\nepoch {}/{}".format(epoch+1,epochs))
            progressBar = Progbar(steps_per_epoch, stateful_metrics=metrics_names)

            # Training step
            self.train(source, target, steps_per_epoch, progressBar, (epoch+1))
            self.seg_model.reset_states()
            self.disc_model.reset_states()

            # validation step
            val_seg_loss = self.validate(val_data)
            self.seg_model.reset_states()
            self.disc_model.reset_states()

            # EarlyStopping operations
            patience, stop_training = self.EarlyStopper(val_seg_loss, patience)
            
            # stop training if the conditions in Early Stopper holds
            if stop_training:
                break 
        # takes the best model with less validation loss.
        self.seg_model.model = tf.keras.models.load_model(self.save_to, compile=False)
