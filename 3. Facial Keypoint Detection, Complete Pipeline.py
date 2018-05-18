
# coding: utf-8

# ## Face and Facial Keypoint detection
# 
# After you've trained a neural network to detect facial keypoints, you can then apply this network to *any* image that includes faces. The neural network expects a Tensor of a certain size as input and, so, to detect any face, you'll first have to do some pre-processing.
# 
# 1. Detect all the faces in an image using a face detector (we'll be using a Haar Cascade detector in this notebook).
# 2. Pre-process those face images so that they are grayscale, and transformed to a Tensor of the input size that your net expects. This step will be similar to the `data_transform` you created and applied in Notebook 2, whose job was tp rescale, normalize, and turn any iimage into a Tensor to be accepted as input to your CNN.
# 3. Use your trained model to detect facial keypoints on the image.
# 
# ---

# In the next python cell we load in required libraries for this section of the project.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# <div class="alert alert-info">**Note:** Workspaces try to preserve your available GPU time by closing the connection after 30 minutes of inactivity (including while training!). Use the code snippet below to keep your workspace alive during training.
# </div>

# In[2]:


import requests
response = requests.request("GET", "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", headers={"Metadata-Flavor":"Google"})
token = response.text

# add the next line to run in each iteration of your main training loop
# requests.request("POST", "https://nebula.udacity.com/api/v1/remote/keep-alive", headers={'Authorization': "STAR " + token})


# #### Select an image 
# 
# Select an image to perform facial keypoint detection on; you can select any image of faces in the `images/` directory.

# In[3]:


import cv2
# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)


# ## Detect all faces in an image
# 
# Next, you'll use one of OpenCV's pre-trained Haar Cascade classifiers, all of which can be found in the `detector_architectures/` directory, to find any faces in your selected image.
# 
# In the code below, we loop over each face in the original image and draw a red square on each face (in a copy of the original image, so as not to modify the original). You can even [add eye detections](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) as an *optional* exercise in using Haar detectors.
# 
# An example of face detection on a variety of images is shown below.
# 
# <img src='images/haar_cascade_ex.png' width=80% height=80%/>
# 

# In[4]:


# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)


# ## Loading in a trained model
# 
# Once you have an image to work with (and, again, you can select any image of faces in the `images/` directory), the next step is to pre-process that image and feed it into your CNN facial keypoint detector.
# 
# First, load your best model by its filename.
# 
# Again **please delete any checkpoints in the `saved_models` directory before you submit this project.** 

# In[5]:


import torch
from models import Net

net = Net()

## TODO: load the best saved model parameters (by your path name)
## You'll need to un-comment the line below and add the correct name for *your* saved model
# net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))

net.load_state_dict(torch.load('saved_models/FacialKeyPnts_TrainedModelUpdated.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()


# ## Keypoint detection
# 
# Now, we'll loop over each detected face in an image (again!) only this time, you'll transform those faces in Tensors that your CNN can accept as input images.
# 
# ### TODO: Transform each detected face into an input Tensor
# 
# You'll need to perform the following steps for each detected face:
# 1. Convert the face from RGB to grayscale
# 2. Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
# 3. Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
# 4. Reshape the numpy image into a torch image.
# 
# You may find it useful to consult to transformation code in `data_load.py` to help you perform these processing steps.
# 
# 
# ### TODO: Detect and display the predicted keypoints
# 
# After each face has been appropriately converted into an input Tensor for your network to see as input, you'll wrap that Tensor in a Variable() and can apply your `net` to each face. The ouput should be the predicted the facial keypoints. These keypoints will need to be "un-normalized" for display, and you may find it helpful to write a helper function like `show_keypoints`. You should end up with an image like the following with facial keypoints that closely match the facial features on each individual face:
# 
# <img src='images/michelle_detected.png' width=30% height=30%/>
# 
# 
# 

# In[6]:


from torch.autograd import Variable

image_copy = np.copy(image)

# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
  #  roi = image_copy[y:y+h, x:x+w]
    
  #  roi=image_copy[y-30:y+h+20, x-30:x+w+20]

    roi=image_copy[y-60:y+h+60, x-60:x+w+60]
    
    ## TODO: Convert the face region from RGB to grayscale    
    gray_image = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]    
    norm_gray = gray_image/255.0
    
    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)    
    norm_gray_rescaled = cv2.resize(norm_gray, (224, 224))
      
  ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
                                                                    
    norm_gray_reshaped = norm_gray_rescaled.reshape(norm_gray_rescaled.shape[0], norm_gray_rescaled.shape[1], 1) 
    norm_gray_reshaped = torch.from_numpy(norm_gray_reshaped.transpose((2, 0, 1))) # reshaping to C H W
    norm_gray_reshaped = norm_gray_reshaped.unsqueeze(0).float() 
   # print(type(norm_gray_reshaped))
     
    ## TODO: Make facial keypoint predictions using your loaded, trained network 
        
    ## wrap each face region in a Variable and perform a forward pass to get the predicted facial keypoints
    
    norm_gray_reshaped = Variable(norm_gray_reshaped)
    norm_gray_reshaped = norm_gray_reshaped.type(torch.FloatTensor)
    output_pts = net.forward(norm_gray_reshaped)

    ## TODO: Display each detected face and the corresponding keypoints      
    
    predicted_key_pts = output_pts.data
    
    predicted_key_pts = predicted_key_pts.numpy()
    
    # undo normalization of keypoints  
    predicted_key_pts = predicted_key_pts*50.0+100   
    
    predicted_key_pts = np.reshape(predicted_key_pts, (68,2))
    
 #   predicted_key_pts[0,:] = predicted_key_pts[0,:] + y - 60
 #   predicted_key_pts[1,:] = predicted_key_pts[1,:] + x - 60       
 #   print(np.squeeze(norm_gray_reshaped.data.numpy()).shape)
    
    fig = plt.figure(figsize=(6,6))

    plt.imshow(np.squeeze(norm_gray_reshaped.data.numpy()), cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
  #  plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=5, marker='.', c='m')
    plt.axis('off')

plt.show()    


# In[7]:


##K-MEANS Clustering Algorithm Implementation## 

Kmeans_image = cv2.imread('images/mona_lisa.jpg')

Kmeans_image = cv2.cvtColor(Kmeans_image, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(9,9))
plt.imshow(Kmeans_image)


# In[8]:



pixel_vals = Kmeans_image.reshape((-1,3))

pixel_vals = np.float32(pixel_vals)


# In[9]:


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)

k = 5
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((Kmeans_image.shape))
labels_reshape = labels.reshape(Kmeans_image.shape[0], Kmeans_image.shape[1])

plt.imshow(segmented_image)


# In[12]:


plt.imshow(labels_reshape==4, cmap='gray')

#(this segment visualizes the Face and hands and basically the skin color of the Lady in the painting)           


# In[14]:


plt.imshow(labels_reshape==2, cmap='gray') 
#(this segment visualizes the Dark Hair of the Lady and Dark Robee the Lady  is wearing in the painting)


# In[17]:


cluster = 4 # the second cluster

masked_image = np.copy(Kmeans_image)

masked_image[labels_reshape != cluster] = [0, 0, 0]

plt.imshow(masked_image)


# In[18]:


cluster = 2 # the fourth cluster

masked_image = np.copy(Kmeans_image)

masked_image[labels_reshape != cluster] = [0, 0, 0]

plt.imshow(masked_image)

