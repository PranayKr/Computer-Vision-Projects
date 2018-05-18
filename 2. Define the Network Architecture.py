
# coding: utf-8

# ## Define the Convolutional Neural Network
# 
# After you've looked at the data you're working with and, in this case, know the shapes of the images and of the keypoints, you are ready to define a convolutional neural network that can *learn* from this data.
# 
# In this notebook and in `models.py`, you will:
# 1. Define a CNN with images as input and keypoints as output
# 2. Construct the transformed FaceKeypointsDataset, just as before
# 3. Train the CNN on the training data, tracking loss
# 4. See how the trained model performs on test data
# 5. If necessary, modify the CNN structure and model hyperparameters, so that it performs *well* **\***
# 
# **\*** What does *well* mean?
# 
# "Well" means that the model's loss decreases during training **and**, when applied to test image data, the model produces keypoints that closely match the true keypoints of each face. And you'll see examples of this later in the notebook.
# 
# ---
# 

# ## CNN Architecture
# 
# Recall that CNN's are defined by a few types of layers:
# * Convolutional layers
# * Maxpooling layers
# * Fully-connected layers
# 
# You are required to use the above layers and encouraged to add multiple convolutional layers and things like dropout layers that may prevent overfitting. You are also encouraged to look at literature on keypoint detection, such as [this paper](https://arxiv.org/pdf/1710.00977.pdf), to help you determine the structure of your network.
# 
# 
# ### TODO: Define your model in the provided file `models.py` file
# 
# This file is mostly empty but contains the expected name and some TODO's for creating your model.
# 
# ---

# ## PyTorch Neural Nets
# 
# To define a neural network in PyTorch, you define the layers of a model in the function `__init__` and define the feedforward behavior of a network that employs those initialized layers in the function `forward`, which takes in an input image tensor, `x`. The structure of this Net class is shown below and left for you to fill in.
# 
# Note: During training, PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior and using autograd to calculate the update to the weights in the network.
# 
# #### Define the Layers in ` __init__`
# As a reminder, a conv/pool layer may be defined like this (in `__init__`):
# ```
# # 1 input image channel (for grayscale images), 32 output channels/feature maps, 3x3 square convolution kernel
# self.conv1 = nn.Conv2d(1, 32, 3)
# 
# # maxpool that uses a square window of kernel_size=2, stride=2
# self.pool = nn.MaxPool2d(2, 2)      
# ```
# 
# #### Refer to Layers in `forward`
# Then referred to in the `forward` function like this, in which the conv1 layer has a ReLu activation applied to it before maxpooling is applied:
# ```
# x = self.pool(F.relu(self.conv1(x)))
# ```
# 
# Best practice is to place any layers whose weights will change during the training process in `__init__` and refer to them in the `forward` function; any layers or functions that always behave in the same way, such as a pre-defined activation function, should appear *only* in the `forward` function.

# #### Why models.py
# 
# You are tasked with defining the network in the `models.py` file so that any models you define can be saved and loaded by name in different notebooks in this project directory. For example, by defining a CNN class called `Net` in `models.py`, you can then create that same architecture in this and other notebooks by simply importing the class and instantiating a model:
# ```
#     from models import Net
#     net = Net()
# ```

# In[1]:


# load the data if you need to; if you have already loaded the data, you may comment this cell out
# -- DO NOT CHANGE THIS CELL -- #
get_ipython().system('mkdir /data')
get_ipython().system('wget -P /data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip')
get_ipython().system('unzip /data/train-test-data.zip -d /data')


# <div class="alert alert-info">**Note:** Workspaces preserve your available GPU time by closing the connection after 30 minutes of inactivity (including inactivity while training!). Use the code snippet below to keep your workspace alive during training.
# </div>

# In[2]:


import requests
response = requests.request("GET", "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", headers={"Metadata-Flavor":"Google"})
token = response.text

# add the next line to run in each iteration of your main training loop
# requests.request("POST", "https://nebula.udacity.com/api/v1/remote/keep-alive", headers={'Authorization': "STAR " + token})


# In[3]:


# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

# watch for any changes in model.py, if it changes, re-load it automatically
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


## TODO: Define the Net in models.py

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from models import Net

net = Net()
print(net)

net.conv1.bias.data.fill_(0);
net.conv1.weight.data.normal_(std=0.01);

net.conv2.bias.data.fill_(0);
net.conv2.weight.data.normal_(std=0.01);


net.conv3.bias.data.fill_(0);
net.conv3.weight.data.normal_(std=0.01);

net.conv4.bias.data.fill_(0);
net.conv4.weight.data.normal_(std=0.01);


net.fc1.bias.data.fill_(0);
net.fc1.weight.data.normal_(std=0.01);

net.fc2.bias.data.fill_(0);
net.fc2.weight.data.normal_(std=0.01);


# ## Transform the dataset 
# 
# To prepare for training, create a transformed dataset of images and keypoints.
# 
# ### TODO: Define a data transform
# 
# In PyTorch, a convolutional neural network expects a torch image of a consistent size as input. For efficient training, and so your model's loss does not blow up during training, it is also suggested that you normalize the input images and keypoints. The necessary transforms have been defined in `data_load.py` and you **do not** need to modify these; take a look at this file (you'll see the same transforms that were defined and applied in Notebook 1).
# 
# To define the data transform below, use a [composition](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms) of:
# 1. Rescaling and/or cropping the data, such that you are left with a square image (the suggested size is 224x224px)
# 2. Normalizing the images and keypoints; turning each RGB image into a grayscale image with a color range of [0, 1] and transforming the given keypoints into a range of [-1, 1]
# 3. Turning these images and keypoints into Tensors
# 
# These transformations have been defined in `data_load.py`, but it's up to you to call them and create a `data_transform` below. **This transform will be applied to the training data and, later, the test data**. It will change how you go about displaying these images and keypoints, but these steps are essential for efficient training.
# 
# As a note, should you want to perform data augmentation (which is optional in this project), and randomly rotate or shift these images, a square image size will be useful; rotating a 224x224 image by 90 degrees will result in the same shape of output.

# In[5]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(256), 
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'


# In[6]:


# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',
                                             root_dir='/data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


# ## Batching and loading data
# 
# Next, having defined the transformed dataset, we can use PyTorch's DataLoader class to load the training data in batches of whatever size as well as to shuffle the data for training the model. You can read more about the parameters of the DataLoader, in [this documentation](http://pytorch.org/docs/master/data.html).
# 
# #### Batch size
# Decide on a good batch size for training your model. Try both small and large batch sizes and note how the loss decreases as the model trains.
# 
# **Note for Windows users**: Please change the `num_workers` to 0 or you may face some issues with your DataLoader failing.

# In[7]:


# load training data in batches
batch_size = 10

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)


# ## Before training
# 
# Take a look at how this model performs before it trains. You should see that the keypoints it predicts start off in one spot and don't match the keypoints on a face at all! It's interesting to visualize this behavior so that you can compare it to the model after training and see how the model has improved.
# 
# #### Load in the test dataset
# 
# The test dataset is one that this model has *not* seen before, meaning it has not trained with these images. We'll load in this test data and before and after training, see how your model performs on this set!
# 
# To visualize this test data, we have to go through some un-transformation steps to turn our images into python images from tensors and to turn our keypoints back into a recognizable range. 

# In[8]:


# load in the test data, using the dataset class
# AND apply the data_transform you defined above

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/data/test_frames_keypoints.csv',
                                             root_dir='/data/test/',
                                             transform=data_transform)



# In[9]:


# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)


# ## Apply the model on a test sample
# 
# To test the model on a test sample of data, you have to follow these steps:
# 1. Extract the image and ground truth keypoints from a sample
# 2. Wrap the image in a Variable, so that the net can process it as input and track how it changes as the image moves through the network.
# 3. Make sure the image is a FloatTensor, which the model expects.
# 4. Forward pass the image through the net to get the predicted, output keypoints.
# 
# This function test how the network performs on the first batch of test data. It returns the images, the transformed images, the predicted keypoints (produced by the model), and the ground truth keypoints.

# In[10]:


# test the model on a batch of test images

def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']
        
        # wrap images in a torch Variable
        # key_pts do not need to be wrapped until they are used for training
        images = Variable(images)

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts
            


# #### Debugging tips
# 
# If you get a size or dimension error here, make sure that your network outputs the expected number of keypoints! Or if you get a Tensor type error, look into changing the above code that casts the data into float types: `images = images.type(torch.FloatTensor)`.

# In[11]:


# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


# ## Visualize the predicted keypoints
# 
# Once we've had the model produce some predicted output keypoints, we can visualize these points in a way that's similar to how we've displayed this data before, only this time, we have to "un-transform" the image/keypoint data to display it.
# 
# Note that I've defined a *new* function, `show_all_keypoints` that displays a grayscale image, its predicted keypoints and its ground truth keypoints (if provided).

# In[12]:


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# #### Un-transformation
# 
# Next, you'll see a helper function. `visualize_output` that takes in a batch of images, predicted keypoints, and ground truth keypoints and displays a set of those images and their true/predicted keypoints.
# 
# This function's main role is to take batches of image and keypoint data (the input and output of your CNN), and transform them into numpy images and un-normalized keypoints (x, y) for normal display. The un-transformation process turns keypoints and images into numpy arrays from Tensors *and* it undoes the keypoint normalization done in the Normalize() transform; it's assumed that you applied these transformations when you loaded your test data.

# In[13]:


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()
    
# call it
visualize_output(test_images, test_outputs, gt_pts)


# ## Training
# 
# #### Loss function
# Training a network to predict keypoints is different than training a network to predict a class; instead of outputting a distribution of classes and using cross entropy loss, you may want to choose a loss function that is suited for regression, which directly compares a predicted value and target value. Read about the various kinds of loss functions (like MSE or L1/SmoothL1 loss) in [this documentation](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html).
# 
# ### TODO: Define the loss and optimization
# 
# Next, you'll define how the model will train by deciding on the loss function and optimizer.
# 
# ---

# In[14]:


## TODO: Define the loss and optimization
import torch.optim as optim

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(net.parameters(), lr=0.001)


# ## Training and Initial Observation
# 
# Now, you'll train on your batched training data from `train_loader` for a number of epochs. 
# 
# To quickly observe how your model is training and decide on whether or not you should modify it's structure or hyperparameters, you're encouraged to start off with just one or two epochs at first. As you train, note how your the model's loss behaves over time: does it decrease quickly at first and then slow down? Does it take a while to decrease in the first place? What happens if you change the batch size of your training data or modify your loss function? etc. 
# 
# Use these initial observations to make changes to your model and decide on the best architecture before you train for many epochs and create a final model.

# In[15]:


def train_net(n_epochs):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        requests.request("POST", "https://nebula.udacity.com/api/v1/remote/keep-alive", headers={'Authorization': "STAR " + token})
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            
            # wrap them in a torch Variable
            images, key_pts = Variable(images), Variable(key_pts)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.data[0]
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                running_loss = 0.0

    print('Finished Training')


# In[16]:


# train your network
n_epochs = 2 # start small, and increase when you've decided on your model structure and hyperparams

train_net(n_epochs)


# ## Test data
# 
# See how your model performs on previously unseen, test data. We've already loaded and transformed this data, similar to the training data. Next, run your trained model on these images to see what kind of keypoints are produced. You should be able to see if your model is fitting each new face it sees, if the points are distributed randomly, or if the points have actually overfitted the training data and do not generalize.

# In[17]:


# get a sample of test data again
test_images, test_outputs, gt_pts = net_sample_output()

print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


# In[18]:


## TODO: visualize your test output
# you can use the same function as before, by un-commenting the line below:

visualize_output(test_images, test_outputs, gt_pts)


# Once you've found a good model (or two), save your model so you can load it and use it later!
# 
# Save your models but please **delete any checkpoints and saved models before you submit your project** otherwise your workspace may be too large to submit.

# In[19]:


## TODO: change the name to something uniqe for each new model
model_dir = 'saved_models/'
model_name = 'FacialKeyPnts_TrainedModelUpdated.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)


# In[20]:


net = Net()

net.load_state_dict(torch.load('saved_models/FacialKeyPnts_TrainedModelUpdated.pt'))

print(net)


# After you've trained a well-performing model, answer the following questions so that we have some insight into your training and architecture selection process. Answering all questions is required to pass this project.

# ### Question 1: What optimization and loss functions did you choose and why?
# 

# 
# **Answer**: 1)I used Adam Optimizer function for optimization whuch is basically an extension of Stochastic Gradient Descent Optimizer fumction .The term Adam is derived from Adaptive moment Estimation. Adam Optimizee combines advantages of 
# a)Adaptive Gradient Algorithm(AdaGrad) :Improves performance on problems with Sparse Gradients (for e.g. Natural Language Proceesing and Computer Vision Problem Statements) by maintaining aper-parameter learning Rate 
# and 
# b)Root Mean Square Propagation(RMSProp)  :improves performance on problems characterised as Noisy and Dynamic (Non-Stationary) 
# by maintaining aper-parameter learning Rate that are adapted based on the average of recent magnitudes of the gradients for the weight
# 
# 2) I used SmoothL1Loss Function which is especially suited for regression problems like the current one involved in trying to detect/predict the (x,y) coordinates of Facial Keypoints . SmoothL1Loss Function is less sensitive to outliers than the 'MSELoss' function
# and in some cases prevents exploding gradients

# ### Question 2: What kind of network architecture did you start with and how did it change as you tried different architectures? Did you decide to add more convolutional layers or any layers to avoid overfitting the data?

# **Answer**: Initially I started with 2 convolutional layers each followed by Maxpooling layers and then further stacked up 2 fully connected layers . To prevent Overfitting I had initially added a dropout layer between the convolutional layers and a dropout between fully connected layers but later on training the model it seemed that after the first epoch the model was not learning but kept on fluctuating around the already achieved learning state ... Hence instead of using Dropout layers in the model architecture I used a 2D-batchnormalization layer after the second Convolutional Layer and another 1D-BtachNormalization layer after the first fully connected layer and on training found that the new model was learning more quickly and efficiently at the same learning rate 
# 
# Besides I also initialized the weights and biases of the 2 Convolutional layers and 2 Fully Connected layers setting the value of biases to be 0 and weights to normalized values over a standard deviation of 0.01

# ### Question 3: How did you decide on the number of epochs and batch_size to train your model?

# **Answer**: 1)Initially I had attempted to train the model with batch-size of 20 but found that it was taking too much time to even train the first set of 20 batches ... Hence I changed the batch-size to 10 after which the training of the model progressed visibly faster 
# 2) I have only trained the model for 2 Epochs because there is issue with internet connectivity at my place and very frequently the Internet gets disconnected ... Due to this issue of Inconsistent network connectivity I could not proceed with training the model for a large number of Epochs 
# Given ideal scenario I guess I would have chosen to train the model fpr around 20 Epochs and then check the results . If the results would not have been as expected I would have significantly increased/decreased the number of Epochs for the model to be trained as would have been appropriate/suitable to achive the proper results
# 

# ## Feature Visualization
# 
# Sometimes, neural networks are thought of as a black box, given some input, they learn to produce some output. CNN's are actually learning to recognize a variety of spatial patterns and you can visualize what each convolutional layer has been trained to recognize by looking at the weights that make up each convolutional kernel and applying those one at a time to a sample image. This technique is called feature visualization and it's useful for understanding the inner workings of a CNN.

# In the cell below, you can see how to extract a single filter (by index) from your first convolutional layer. The filter should appear as a grayscale grid.

# In[108]:


# Get the weights in the first conv layer, "conv1"
# if necessary, change this to reflect the name of your first conv layer
weights1 = net.conv1.weight.data

w = weights1.numpy()

filter_index = 24

print(w[filter_index][0])
print(w[filter_index][0].shape)

# display the filter weights
plt.imshow(w[filter_index][0], cmap='gray')


# ## Feature maps
# 
# Each CNN has at least one convolutional layer that is composed of stacked filters (also known as convolutional kernels). As a CNN trains, it learns what weights to include in it's convolutional kernels and when these kernels are applied to some input image, they produce a set of **feature maps**. So, feature maps are just sets of filtered images; they are the images produced by applying a convolutional kernel to an input image. These maps show us the features that the different layers of the neural network learn to extract. For example, you might imagine a convolutional kernel that detects the vertical edges of a face or another one that detects the corners of eyes. You can see what kind of features each of these kernels detects by applying them to an image. One such example is shown below; from the way it brings out the lines in an the image, you might characterize this as an edge detection filter.
# 
# <img src='images/feature_map_ex.png' width=50% height=50%/>
# 
# 
# Next, choose a test image and filter it with one of the convolutional kernels in your trained CNN; look at the filtered output to get an idea what that particular kernel detects.
# 
# ### TODO: Filter an image to see the effect of a convolutional kernel
# ---

# In[111]:


##TODO: load in and display any image from the transformed test dataset
import cv2


index = 2

image = test_images[index].data   # get the image from it's Variable wrapper
image = image.numpy()   # convert to numpy array from a Tensor
image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

image = np.squeeze(image)

plt.imshow(image, cmap='gray')


filtered_img = cv2.filter2D(image, -1, w[24][0])
plt.imshow(filtered_img, cmap='gray')

plt.show()


# ### Question 4: Choose one filter from your trained CNN and apply it to a test image; what purpose do you think it plays? What kind of feature do you think it detects?
# 

# **Answer**: The image kernel/filter convolved with the image above detects Horizontal Lines in the image as is evident by the more pronounced visibility of the gradients along the x-axis . 
# 
# Most probably hence I can infer that a Sobel Y Filter is being used here 
# 

# ---
# ## Moving on!
# 
# Now that you've defined and trained your model (and saved the best model), you are ready to move on to the last notebook, which combines a face detector with your saved model to create a facial keypoint detection system that can predict the keypoints on *any* face in an image!
