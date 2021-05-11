from Preprocessing import SupervisedLearning, DomainAdaptation
from Models import Models, Discriminator
from ModelTester import testNetwork
from SupervisedLearning import SupervisedLearner
from DomainAdaptation import DomainAdaptor
from Utils import plotGraph
import tensorflow as tf

DIR = '/data/'

# 1: Supervised Learning, 2: Domain Adaptation
technique =  2

# 1: UNet, 2: FCN (with VGG-16), 3: PSPNet, 4: lightweight
architecture = 3

# Batch size for both train and test sets.
BATCH_SIZE = 16

# Note: The size should be chosen such that PSPNet does not produce error; the size of output of the 4th conv layer should be divisable by 6 because of pyramid pooling module. 
SHAPE = (256, 256, 3) 

# Possible models are chosen based on the variable 'architecture' above and their names are determined for being saved.
seg_model = Models(shape=SHAPE)
if architecture==1: # UNet
       seg_model.UNet()
       save_to = DIR + 'LoadedModels/' + 'UNet' + '.h5'
elif architecture==2: #FCN-8s with VGG-16
       seg_model.FCN()
       save_to = DIR + 'LoadedModels/' + 'FCN' + '.h5'
elif architecture==3: #PSPNet
       seg_model.PSPNet()
       save_to = DIR + 'LoadedModels/' + 'PSPNet' + '.h5'
elif architecture==4: #lightweight
       seg_model.lightweight()
       save_to = DIR + 'LoadedModels/' + 'lightweight' + '.h5'
else :
       raise ValueError("\nNo correct model has been specified! Please, choose appropreate 'architecture' above.\n")


if (technique==1):
       # Directory of Tampere-WaterSeg dataset
       # splitter type: left represents data used for training while right of '/' stands for data used for testing. 
       # e.g., a/a means all/all, c/od means channel/open+dock as in Table 1. in WaterSeg paper.    
       directory = '/data/Tampere-WaterSeg'
       SPLITTER_TYPE = 'a/a'


       # Getting train, validation and test images along with corresponding ground truth masks. 
       train_data, val_images, test_images, steps_per_epoch, test_names = SupervisedLearning(
                                                 directory=directory, shape=SHAPE,batch_size=BATCH_SIZE, split_type = SPLITTER_TYPE, architecture=architecture
                                                 ).generateImages()

       # Batching validation data.
       val_data = tf.data.Dataset.from_tensor_slices((val_images[0], val_images[1])).batch(BATCH_SIZE)
       # Batching test data. 
       test_data = tf.data.Dataset.from_tensor_slices((test_images[0], test_images[1])).batch(BATCH_SIZE)

       # prepare supervised learning method
       trainer = SupervisedLearner(25, save_to, seg_model, architecture)

       # train the model 
       trainer.fit(train_data, val_data, epochs=100, steps_per_epoch=steps_per_epoch)
       
       # The graph of f1_score: train vs validation
       plotGraph(ylabel='F1 score', xlabel='Epoch', save_to=DIR + 'Graphs/f1_score.png', index=5, model=seg_model, location='lower right')

       # The graph of loss: train vs validation
       plotGraph(ylabel='Loss', xlabel='Epoch', save_to= DIR + 'Graphs/loss.png', index=0, model=seg_model, location='upper right')

       # Extract F1 scores based on states (open, dock, channel); Find images with min and max F1 score, save these images.
       testNetwork(test_data, test_images, seg_model, test_names, architecture)

elif technique==2:
       #model = tf.keras.models.load_model("Best_Worst_Outputs/PSPNet.h5", compile=False)
       #seg_model.model.set_weights(model.get_weights())
       # target directory statically represents TampereWaterSeg dataset.
       # source directory might change based on chosen parameters below   
       target_directory = '/data/Tampere-WaterSeg'
       source_directory = 12 # 1. MaSTr1325, 2. WaterDataset 3. any number except 1 and 2 takes both. 

       # Getting train (from source and target (unlabeled) domain) and test images along with corresponding ground truth masks. 
       source_train, target_train, val_images, test_images, steps_per_epoch, test_names = DomainAdaptation( source_dir=source_directory, target_dir=target_directory, 
                                                                                                  shape=SHAPE, batch_size=BATCH_SIZE, architecture=architecture).generateImages()


       # create batch of datas within validation set. 
       val_data = tf.data.Dataset.from_tensor_slices((val_images[0], val_images[1])).batch(BATCH_SIZE)
       # create batch of datas within test set. 
       test_data = tf.data.Dataset.from_tensor_slices((test_images[0], test_images[1])).batch(BATCH_SIZE)

       # define discriminator model
       disc_model = Discriminator(shape=SHAPE)
       
       # prepare Domain Adaptation learning method.
       trainer = DomainAdaptor(25, save_to, seg_model, disc_model, architecture)

       # start training the model
       trainer.fit(source_train, target_train, val_data, epochs=100, steps_per_epoch=steps_per_epoch)
       
       # The graph of f1_score: train vs validation
       plotGraph(ylabel='F1 score', xlabel='Epoch', save_to= DIR + 'Graphs/f1_score.png', index=7, model=seg_model, location='lower right')

       # The graph of loss: train vs validation
       plotGraph(ylabel='Loss', xlabel='Epoch', save_to= DIR + 'Graphs/loss.png', index=0, model=seg_model, location='upper right')

       # The graph of adversarial loss: train vs validation
       plotGraph(ylabel='Adversarial Loss', xlabel='Epoch', save_to=DIR + 'Graphs/adv_loss.png', index=1, model=seg_model, location='upper right')

       # The graph of discriminator loss: train vs validation
       plotGraph(ylabel='Discriminator Loss', xlabel='Epoch', save_to=DIR + 'Graphs/disc_loss.png', index=2, model=seg_model, location='upper right')

       # Extract F1 scores based on states (open, dock, channel); Find images with min and max F1 score, save these images.
       testNetwork(test_data, test_images, seg_model, test_names, architecture)

else :
       raise ValueError("\nPlease, choose a correct parameter for 'technique': 1 stands for Unsupervised Learning, 2 stands for Domain Adaptation method\n")
