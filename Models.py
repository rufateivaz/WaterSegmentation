from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Input, LeakyReLU, BatchNormalization, UpSampling2D, Add, Activation, MaxPooling2D, concatenate
import tensorflow as tf

DIR = '/data/'

# Discriminator architecture which is used to determine input comes either from source or target domain.
class Discriminator():
    def __init__(self, shape):
        # Discriminator model is defined in the constructor of the Discriminator class.
        input_layer = Input(shape=(shape[0], shape[1], 1), name='input_to_discriminator')
        x = input_layer
        coeffs = [1, 2, 4, 8]
        for coeff in coeffs:
            x = Conv2D(filters= 64 * coeff, kernel_size=(4, 4), strides=(2,2), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=1, kernel_size=(4,4), strides=(2, 2), padding='same')(x)
        x = UpSampling2D(size=16, interpolation='bilinear')(x)
        x = Activation('sigmoid')(x)
        self.model = Model(inputs=input_layer, outputs=x, name='Discriminator')

        # this is the learning rate for training the discriminator
        self.lr = 1e-4

        # this is the loss function that is used to compute the BCE loss for the discriminator
        self.loss = tf.keras.losses.BinaryCrossentropy() 

        # these are the metrics, that are used as metrics for both training and validation phases; so as to print the graph in the end.
        self.adv_loss = tf.keras.metrics.BinaryCrossentropy()
        self.disc_loss = tf.keras.metrics.BinaryCrossentropy()

        # this is the optimzer that is used to update weigths of the discriminator
        self.optimizer= tf.keras.optimizers.Adam(self.lr)

        # this is just for printing the model's parameters to see if everything is Ok.
        self.model.summary()
        

    # accumulates adversarial and discriminator losses after each processed batch in validation phase 
    # for calculating the average adversarial or discriminator loss 
    def update_states(self, fake, real, pred):
        self.adv_loss.update_state(fake, pred)
        self.disc_loss.update_state(real, pred)
    
    # resets accumulated discriminator and adversarial losses.
    def reset_states(self):
        self.adv_loss.reset_states()
        self.disc_loss.reset_states()

# This is the layer that is used by all networks : UNet, PSPNet and lightweight
def common_layer(x, filter, size, padding, initializer, using_bias):
    x = Conv2D(filters=filter, kernel_size=(size, size), padding=padding, kernel_initializer=initializer, use_bias=using_bias)(x)  
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)
    return x

# This is a spatial phase that is used by PSPNet for pyramid pooling phase.
def spatial_phase(x, level, padding, initializer, using_bias):
    h = x.shape[1]
    w = x.shape[2]
    strides = (h//level, w//level)
    x = AveragePooling2D(pool_size = strides, strides=strides, padding=padding)(x) 
    x = common_layer(x, 512, 1, padding, initializer, using_bias)
    x = UpSampling2D(size=strides, interpolation='bilinear')(x)
    return x

# This class contains many information. Some necessary information are provided below.
# self.model is the model that is initially none but based on chosen chosen architecture it becomes that network.
# self.shape represents the shape of the input tensor/numpy 
# self.lr stands for the learning rate of the model. 
# self.tp,fp,tn,fn are confusion matrix params that are used for calculating sensitivity, specificity, precision, accuracy and f1 score in utils.py
# train_history and val_history are used for keeping track of losses or metrics for being able to compare train and validation phases later. 
class Models():
    def __init__(self, shape):
        self.model = None
        self.shape = shape
        self.lr = 2.5e-4

        # this is the BCE loss function to compute segmentation loss
        self.loss = tf.keras.losses.BinaryCrossentropy()
    
        # this is the adam optimizer, used for updating weigths. 
        self.optimizer= tf.keras.optimizers.Adam(self.lr)
        # this is used when calculating the validation loss at the end of each epoch.
        self.seg_loss = tf.keras.metrics.BinaryCrossentropy()
        
        # these variables are the binary confusion matrix parameters; they are used for computing the scores (e.g., f1 score) 
        self.tp = tf.keras.metrics.TruePositives(thresholds=0.5)
        self.fp = tf.keras.metrics.FalsePositives(thresholds=0.5)
        self.tn = tf.keras.metrics.TrueNegatives(thresholds=0.5)
        self.fn = tf.keras.metrics.FalseNegatives(thresholds=0.5)

        # these variables are used for taking train and validation scores after each 
        # epoch so as to draw graph in the end to analyze the behaviour of the model in both phases
        self.train_history = [] 
        self.val_history = []

    # resets accumulated segmentation loss and confusion matrix params. 
    def reset_states(self):
        self.seg_loss.reset_states()
        self.tp.reset_states()
        self.tn.reset_states()
        self.fp.reset_states()
        self.fn.reset_states()

    # accumulates segmentation loss and confusion matrix params after each presented batch.
    def update_states(self, y_true, y_pred):
        self.seg_loss.update_state(y_true, y_pred)
        self.tp.update_state(y_true, y_pred)
        self.tn.update_state(y_true, y_pred)
        self.fp.update_state(y_true, y_pred)
        self.fn.update_state(y_true, y_pred)
    
    
    # This model is FCN model, which is used in Tampere-WaterSeg for Tests 1-10, in Table 1. The parameters might differ but the general architecture is this. 
    def FCN(self, nfilters= 64, size=3, strides=2, padding='same', initializer='he_normal', using_bias=False):
        input_layer = Input(shape=self.shape)
        base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=input_layer)

        x = common_layer(base_model.output, filter=4096, size=7, padding=padding, initializer=initializer, using_bias=using_bias)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = common_layer(x, filter=4096, size=1, padding=padding, initializer=initializer, using_bias=using_bias)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Decoder
        x = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal')(x) 
        x = Conv2D(filters=1, kernel_size=(4,4), padding='same', kernel_initializer=initializer, use_bias=False)(UpSampling2D(size=strides, interpolation='bilinear')(x))
        pool4_res = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal')(base_model.get_layer("block4_pool").output) 
        x = Add()([x, pool4_res])
        x = Conv2D(filters=1, kernel_size=(4,4), padding='same', kernel_initializer=initializer, use_bias=False)(UpSampling2D(size=strides, interpolation='bilinear')(x))
        pool3_res = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal')(base_model.get_layer("block3_pool").output)
        x = Add()([x, pool3_res])
        x = Conv2D(filters=1, kernel_size=(16,16), padding='same', kernel_initializer=initializer, use_bias=False)(UpSampling2D(size=8, interpolation='bilinear')(x))
        x = Activation('sigmoid')(x)   

        self.model = Model(inputs=base_model.input, outputs=x, name='FCN-8s')
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file=DIR + 'ModelViews/FCN.png', show_shapes=True)


    # lightweight network proposed by TampereWaterSeg paper. They use their own encoder but the same decoder as in FCN, implemented above.
    def lightweight(self, nfilters=32, size=3, strides=2, padding='same', initializer='he_normal', using_bias=False):
        input_layer = Input(shape=self.shape)
        filters = [1 * nfilters, 2 * nfilters, 4 * nfilters, 8 * nfilters, 16 * nfilters]
        # Encoder
        x = input_layer
        for i in range(len(filters)):
            x = common_layer(x, filters[i], size, padding, initializer, using_bias)
            x = MaxPooling2D((2, 2), strides=strides, name='pool' + str(i+1))(x) 

        x = common_layer(x, filter=4096, size=7, padding=padding, initializer=initializer, using_bias=using_bias)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = common_layer(x, filter=4096, size=1, padding=padding, initializer=initializer, using_bias=using_bias)
        x = tf.keras.layers.Dropout(0.5)(x)

        encoder = Model(inputs=input_layer, outputs=x)

        # Decoder
        x = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal')(x) 
        x = Conv2D(filters=1, kernel_size=(4,4), padding='same', kernel_initializer=initializer, use_bias=False)(UpSampling2D(size=strides, interpolation='bilinear')(x))
        pool4_res = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal')(encoder.get_layer("pool4").output) 
        x = Add()([x, pool4_res])
        x = Conv2D(filters=1, kernel_size=(4,4), padding='same', kernel_initializer=initializer, use_bias=False)(UpSampling2D(size=strides, interpolation='bilinear')(x))
        pool3_res = Conv2D(filters=1, kernel_size=1, kernel_initializer='he_normal')(encoder.get_layer("pool3").output)
        x = Add()([x, pool3_res])
        x = Conv2D(filters=1, kernel_size=(16,16), padding='same', kernel_initializer=initializer, use_bias=False)(UpSampling2D(size=8, interpolation='bilinear')(x))
        x = Activation('sigmoid')(x)   

        self.model = Model(inputs=input_layer, outputs=x, name='lightweight')
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file=DIR + 'ModelViews/lightweight.png', show_shapes=True)

    # PSPNet / Pyramid Scene Parsing Network is build here.
    def PSPNet(self, size=3, padding='same', initializer='he_normal', using_bias=False):
        input_layer = Input(self.shape)
        # built-in base model / backbone resnet50 is used for encoder part of PSPNet 
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)

        # Pyramid pooling phase is determined here.
        l0 = base_model.get_layer('conv3_block4_out').output
        l1 = spatial_phase(l0, 1, padding, initializer, using_bias)
        l2 = spatial_phase(l0, 2, padding, initializer, using_bias)
        l3 = spatial_phase(l0, 4, padding, initializer, using_bias)
        l6 = spatial_phase(l0, 8, padding, initializer, using_bias)
        x = concatenate([l0, l1, l2, l3, l6], axis=3)
        x = common_layer(x, 512, size, padding, initializer, using_bias)

        # Decoder

        x = Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
        x = UpSampling2D(size=8, interpolation='bilinear')(x)
        x = Activation('sigmoid')(x)
        
        self.model = Model(inputs = base_model.input, outputs = x, name='PSPNet')
        self.model.summary()

        tf.keras.utils.plot_model(self.model, to_file=DIR + 'ModelViews/PSPNet.png', show_shapes=True)
        
    # UNet architecture.
    def UNet(self, nfilters=32, size = 3, padding='same', initializer='he_normal', strides=(2,2), using_bias=False):
        input_layer = Input(self.shape)
        filters = [1 * nfilters, 2 * nfilters, 4 * nfilters, 8 * nfilters, 16 * nfilters]
        
        # Build encoder
        x = input_layer
        for i in range(len(filters)):
            x = common_layer(x, filters[i], size, padding, initializer, using_bias)
            x = common_layer(x, filters[i], size, padding, initializer, using_bias)
            if i < len(filters)-1:
                x = MaxPooling2D(pool_size=strides, padding=padding)(x)
        encoder = Model(inputs=input_layer, outputs=x)

        # Build decoder
        for i in reversed(range(len(filters)-1)):
            x = Conv2D(filters=filters[i], kernel_size=strides, padding='same', kernel_initializer=initializer)(UpSampling2D(size=2, interpolation='bilinear')(x))
            x = concatenate([x, encoder.get_layer("activation_" + str(i*2 + 1)).output], axis=3)
            x = common_layer(x, filters[i], size, padding, initializer, using_bias)
            x = common_layer(x, filters[i], size, padding, initializer, using_bias)

        x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', kernel_initializer=initializer)(x)
        x = Activation('sigmoid')(x)
        self.model = Model(inputs=encoder.input, outputs=x, name='UNet')
        self.model.summary()

        tf.keras.utils.plot_model(self.model, to_file=DIR + 'ModelViews/UNet.png', show_shapes=True)
