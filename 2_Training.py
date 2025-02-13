
# coding: utf-8

# # Computer Vision Nanodegree
# 
# ## Project: Image Captioning
# 
# ---
# 
# In this notebook, you will train your CNN-RNN model.  
# 
# You are welcome and encouraged to try out many different architectures and hyperparameters when searching for a good model.
# 
# This does have the potential to make the project quite messy!  Before submitting your project, make sure that you clean up:
# - the code you write in this notebook.  The notebook should describe how to train a single CNN-RNN architecture, corresponding to your final choice of hyperparameters.  You should structure the notebook so that the reviewer can replicate your results by running the code in this notebook.  
# - the output of the code cell in **Step 2**.  The output should show the output obtained when training the model from scratch.
# 
# This notebook **will be graded**.  
# 
# Feel free to use the links below to navigate the notebook:
# - [Step 1](#step1): Training Setup
# - [Step 2](#step2): Train your Model
# - [Step 3](#step3): (Optional) Validate your Model

# <a id='step1'></a>
# ## Step 1: Training Setup
# 
# In this step of the notebook, you will customize the training of your CNN-RNN model by specifying hyperparameters and setting other options that are important to the training procedure.  The values you set now will be used when training your model in **Step 2** below.
# 
# You should only amend blocks of code that are preceded by a `TODO` statement.  **Any code blocks that are not preceded by a `TODO` statement should not be modified**.
# 
# ### Task #1
# 
# Begin by setting the following variables:
# - `batch_size` - the batch size of each training batch.  It is the number of image-caption pairs used to amend the model weights in each training step. 
# - `vocab_threshold` - the minimum word count threshold.  Note that a larger threshold will result in a smaller vocabulary, whereas a smaller threshold will include rarer words and result in a larger vocabulary.  
# - `vocab_from_file` - a Boolean that decides whether to load the vocabulary from file. 
# - `embed_size` - the dimensionality of the image and word embeddings.  
# - `hidden_size` - the number of features in the hidden state of the RNN decoder.  
# - `num_epochs` - the number of epochs to train the model.  We recommend that you set `num_epochs=3`, but feel free to increase or decrease this number as you wish.  [This paper](https://arxiv.org/pdf/1502.03044.pdf) trained a captioning model on a single state-of-the-art GPU for 3 days, but you'll soon see that you can get reasonable results in a matter of a few hours!  (_But of course, if you want your model to compete with current research, you will have to train for much longer._)
# - `save_every` - determines how often to save the model weights.  We recommend that you set `save_every=1`, to save the model weights after each epoch.  This way, after the `i`th epoch, the encoder and decoder weights will be saved in the `models/` folder as `encoder-i.pkl` and `decoder-i.pkl`, respectively.
# - `print_every` - determines how often to print the batch loss to the Jupyter notebook while training.  Note that you **will not** observe a monotonic decrease in the loss function while training - this is perfectly fine and completely expected!  You are encouraged to keep this at its default value of `100` to avoid clogging the notebook, but feel free to change it.
# - `log_file` - the name of the text file containing - for every step - how the loss and perplexity evolved during training.
# 
# If you're not sure where to begin to set some of the values above, you can peruse [this paper](https://arxiv.org/pdf/1502.03044.pdf) and [this paper](https://arxiv.org/pdf/1411.4555.pdf) for useful guidance!  **To avoid spending too long on this notebook**, you are encouraged to consult these suggested research papers to obtain a strong initial guess for which hyperparameters are likely to work best.  Then, train a single model, and proceed to the next notebook (**3_Inference.ipynb**).  If you are unhappy with your performance, you can return to this notebook to tweak the hyperparameters (and/or the architecture in **model.py**) and re-train your model.
# 
# ### Question 1
# 
# **Question:** Describe your CNN-RNN architecture in detail.  With this architecture in mind, how did you select the values of the variables in Task 1?  If you consulted a research paper detailing a successful implementation of an image captioning model, please provide the reference.
# 
# **Answer:** For implementation of the given Image Captioning Neural Network problem statement:
# 1) I used a SEQ2SEQ model having an a)Encoder CNN and b)Decoder RNN
# 
#    The Encoder CNN (pretrained ResNet-50 ConvNet) is trained on open source COCO Dataset of Images of around 90
#    different Objects 
#    
#    The Decoder RNN is trained on the captions associated wuth the images present as part of the COCO Datset
# 
#    a)Encoder CNN Architecture : 
#    I used pretrained ResNet-50 CNN model as feature extractor from input images by using its
#    pretrained weights and removing the last fully connected Neural Net Layer from the ResNet-50 Stack which is basically 
#    a softmax classifier for classifyung a given image 
#    So now the pretrained ResNet=50 Conv Neural Net is simply encoding the contents of an image into a smaller feature vector
#    and hence called encoder part of the Seq2Seq Model being built 
#    Before providing this feature vector output of the ConvNet to the Decoder RNN it needs to be processed .
#    For that I added an additional Linear Layer on top of the pretrained ConvNet called the embedding layer with number of imput
#    neurons (input size) equal to the number of features extracted by the filter kernels used in the ResNet-50 ConvNet
#    and Output size as the Embedding_Size parameter which has been set as 256 
#    Also I applied Batch-Normalization over the Untrained  Embedding Layer before sending it as input to Decoder RNN
#    
#    b)Decoder RNN Architecture :
#    The first layer is an embedding layer having number of inputs as Vocab_size and outputs as ebedding size.
#    The embedding layer basically converts eaxh word in a caption(vocabulary) to a vector which is then fed to the 
#    LSTM cells Layer along with the feature vector extracted from the Encoder CNN by concatenating both
#    the vectors
#    The Embeddings vector is created bt first removing the last <END> token from the captions 
#    before concatenating with the feature vector 
#    I have used 3 LSTM layers stacked upon each other each having dropout parameter set to 0.4 to 
#    avoid over-fitting
#     
#    Finally the Output of the Decoder RNN is calculated by passing the output of the LSTM layer as input 
#    to a fully connected linear layer 
#    
#    I selected the values of embed_sixe and hidden_size parameter by inferences drawn from going through the 
#    researcg papaers : 1) Show and Tell: A Neural Image Caption Generator (https://arxiv.org/pdf/1411.4555.pdf)
#                       2) Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
#                          (https://arxiv.org/pdf/1502.03044.pdf)
#                          
#                          
#    I inferred that to prevent overfitting and getting more accurate results
#    Deeper the Neural Net Model the better/more accurate would be the results 
#    Hence I chose Emded_size as 256 and Hidden_Size as 512 
#    number of LSTM Cells Layes as 3 
#    and Dropout value of LSTM layers as 0.4 
#    BatchNormalization was also integrated in the Encoder CNN Model for getting better results 
#    and boosting the performance of the SEQ2SEQ model being built
#    
#    I set the Vocab_threshold value to 5 as was suggested
#    
#    I set the number of epochs to 4 for better more accurate results on completion of training
#    
#   
# ### (Optional) Task #2
# 
# Note that we have provided a recommended image transform `transform_train` for pre-processing the training images, but you are welcome (and encouraged!) to modify it as you wish.  When modifying this transform, keep in mind that:
# - the images in the dataset have varying heights and widths, and 
# - if using a pre-trained model, you must perform the corresponding appropriate normalization.
# 
# ### Question 2
# 
# **Question:** How did you select the transform in `transform_train`?  If you left the transform at its provided value, why do you think that it is a good choice for your CNN architecture?
# 
# **Answer:** The values of the parameters in the data_transform functionality are pretty standard and I had used 
# similar values in the First Project (Facual Keypoint detection) as well successfully
# Besides that the Data_trnsform functionality provided has taken care of applying Data Augmentation on the input images
# which leads to better training and performance of the Model being created by making the model more generic in nature 
# of predicting captions on very differnt images it has not been trained on , 
# Resizing and cropping all train images and test images to same resolution which helps the model
# to extract features more generic over a vast collection of images,
# Transforming the Images to Pytorch Tensor as thats the datatype in which the feature vector to be provided to the DecoderRNN
# is expected,
# amd taking care of Normalizaing all the Image Data to a specific shape and value for the Pretrained ConvNET model
# (ResNet-50)  being used here for building Encoder CNN
# 
# 
# 
# ### Task #3
# 
# Next, you will specify a Python list containing the learnable parameters of the model.  For instance, if you decide to make all weights in the decoder trainable, but only want to train the weights in the embedding layer of the encoder, then you should set `params` to something like:
# ```
# params = list(decoder.parameters()) + list(encoder.embed.parameters()) 
# ```
# 
# ### Question 3
# 
# **Question:** How did you select the trainable parameters of your architecture?  Why do you think this is a good choice?
# 
# **Answer:** I selected all the parameters of the Decoder RNN as trainable but only the parameters of Embedding Layer and Batch Normalization Layer of Encoder CNN as trainable because the EnCoder CNN is already using the pretrained weights of the pre-trained ConvNet Model (ResNET-50) but the Embedding layer and Batch-Normalization(on top of Embedding Layer) was added by me on top of the pretrained ConvNET layers and was not originally a part of the ResNet-50 architecture and hence the Embedding and Batch-Normalization layers of the Encoder CNN need to be trained 
# 
# This implementation of the Encoder is using the concept of Transfer Learning to build the entire model 
# rather than creating a model from scratch having all untrained weights which will take a lot of time to implement
# and train before becoming a state of the art solution
# 
# 
# 
# ### Task #4
# 
# Finally, you will select an [optimizer](http://pytorch.org/docs/master/optim.html#torch.optim.Optimizer).
# 
# ### Question 4
# 
# **Question:** How did you select the optimizer used to train your model?
# 
# **Answer:** I selected Adam Optimizer to train the model as was mentioned in the research paper for training on MS-COCO Dataset:
# "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention(https://arxiv.org/pdf/1502.03044.pdf)"
# 
# I used Adam Optimizer with the learning rate value set to 0.001 
# 
# I used Adam Optimizer function for optimization whuch is basically an extension of Stochastic Gradient Descent Optimizer function .The term Adam is derived from Adaptive moment Estimation. Adam Optimizer combines advantages of a)Adaptive Gradient Algorithm(AdaGrad) :Improves performance on problems with Sparse Gradients (for e.g. Natural Language Proceesing and Computer Vision Problem Statements) by maintaining a per-parameter learning Rate and b)Root Mean Square Propagation(RMSProp) :improves performance on problems characterised as Noisy and Dynamic (Non-Stationary) by maintaining a per-parameter learning Rate that are adapted based on the average of recent magnitudes of the gradients for the weight

# In[2]:


import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import math


## TODO #1: Select appropriate values for the Python variables below.
batch_size = 128          # batch size
vocab_threshold = 5        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
embed_size = 256           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_epochs = 4             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# (Optional) TODO #2: Amend the image transform below.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# TODO #3: Specify the learnable parameters of the model.
#params = list(decoder.parameters()) + list(encoder.embed.parameters())
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# TODO #4: Define the optimizer.
optimizer = torch.optim.Adam(params, lr=0.001)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)


# <a id='step2'></a>
# ## Step 2: Train your Model
# 
# Once you have executed the code cell in **Step 1**, the training procedure below should run without issue.  
# 
# It is completely fine to leave the code cell below as-is without modifications to train your model.  However, if you would like to modify the code used to train the model below, you must ensure that your changes are easily parsed by your reviewer.  In other words, make sure to provide appropriate comments to describe how your code works!  
# 
# You may find it useful to load saved weights to resume training.  In that case, note the names of the files containing the encoder and decoder weights that you'd like to load (`encoder_file` and `decoder_file`).  Then you can load the weights by using the lines below:
# 
# ```python
# # Load pre-trained weights before resuming training.
# encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
# decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))
# ```
# 
# While trying out parameters, make sure to take extensive notes and record the settings that you used in your various training runs.  In particular, you don't want to encounter a situation where you've trained a model for several hours but can't remember what settings you used :).
# 
# ### A Note on Tuning Hyperparameters
# 
# To figure out how well your model is doing, you can look at how the training loss and perplexity evolve during training - and for the purposes of this project, you are encouraged to amend the hyperparameters based on this information.  
# 
# However, this will not tell you if your model is overfitting to the training data, and, unfortunately, overfitting is a problem that is commonly encountered when training image captioning models.  
# 
# For this project, you need not worry about overfitting. **This project does not have strict requirements regarding the performance of your model**, and you just need to demonstrate that your model has learned **_something_** when you generate captions on the test data.  For now, we strongly encourage you to train your model for the suggested 3 epochs without worrying about performance; then, you should immediately transition to the next notebook in the sequence (**3_Inference.ipynb**) to see how your model performs on the test data.  If your model needs to be changed, you can come back to this notebook, amend hyperparameters (if necessary), and re-train the model.
# 
# That said, if you would like to go above and beyond in this project, you can read about some approaches to minimizing overfitting in section 4.3.1 of [this paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7505636).  In the next (optional) step of this notebook, we provide some guidance for assessing the performance on the validation dataset.

# In[3]:


import torch.utils.data as data
import numpy as np
import os
import requests
import time

# Open the training log file.
f = open(log_file, 'w')

old_time = time.time()
response = requests.request("GET", 
                            "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", 
                            headers={"Metadata-Flavor":"Google"})

for epoch in range(1, num_epochs+1):
    
    for i_step in range(1, total_step+1):
        
        if time.time() - old_time > 60:
            old_time = time.time()
            requests.request("POST", 
                             "https://nebula.udacity.com/api/v1/remote/keep-alive", 
                             headers={'Authorization': "STAR " + response.text})
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
            
        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
            
    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

# Close the training log file.
f.close()


# <a id='step3'></a>
# ## Step 3: (Optional) Validate your Model
# 
# To assess potential overfitting, one approach is to assess performance on a validation set.  If you decide to do this **optional** task, you are required to first complete all of the steps in the next notebook in the sequence (**3_Inference.ipynb**); as part of that notebook, you will write and test code (specifically, the `sample` method in the `DecoderRNN` class) that uses your RNN decoder to generate captions.  That code will prove incredibly useful here. 
# 
# If you decide to validate your model, please do not edit the data loader in **data_loader.py**.  Instead, create a new file named **data_loader_val.py** containing the code for obtaining the data loader for the validation data.  You can access:
# - the validation images at filepath `'/opt/cocoapi/images/train2014/'`, and
# - the validation image caption annotation file at filepath `'/opt/cocoapi/annotations/captions_val2014.json'`.
# 
# The suggested approach to validating your model involves creating a json file such as [this one](https://github.com/cocodataset/cocoapi/blob/master/results/captions_val2014_fakecap_results.json) containing your model's predicted captions for the validation images.  Then, you can write your own script or use one that you [find online](https://github.com/tylin/coco-caption) to calculate the BLEU score of your model.  You can read more about the BLEU score, along with other evaluation metrics (such as TEOR and Cider) in section 4.1 of [this paper](https://arxiv.org/pdf/1411.4555.pdf).  For more information about how to use the annotation file, check out the [website](http://cocodataset.org/#download) for the COCO dataset.

# In[4]:


# (Optional) TODO: Validate your model.

