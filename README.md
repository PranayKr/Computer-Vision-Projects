# An Automatic Image Captioning System implemented using Sequence to Sequence Model (having Encoder CNN (Convolutional Neural Network) and Decoder RNN (Recurrent Neural Network)) to automatically generate captions from images.

For implementation of the given Image Captioning Neural Network problem statement: 1) a SEQ2SEQ model having an a)Encoder CNN and b)Decoder RNN was used

The Encoder CNN (pretrained ResNet-50 ConvNet) is trained on open source COCO Dataset of Images of around 90 different Objects

The Decoder RNN is trained on the captions associated with the images present as part of the COCO Datset

a)Encoder CNN Architecture : I used pretrained ResNet-50 CNN model as feature extractor from input images by using its pretrained weights and removing the last fully connected Neural Net Layer from the ResNet-50 Stack which is basically a softmax classifier for classifyung a given image So now the pretrained ResNet=50 Conv Neural Net is simply encoding the contents of an image into a smaller feature vector and hence called encoder part of the Seq2Seq Model being built Before providing this feature vector output of the ConvNet to the Decoder RNN it needs to be processed . For that I added an additional Linear Layer on top of the pretrained ConvNet called the embedding layer with number of imput neurons (input size) equal to the number of features extracted by the filter kernels used in the ResNet-50 ConvNet and Output size as the Embedding_Size parameter which has been set as 256 Also I applied Batch-Normalization over the Untrained Embedding Layer before sending it as input to Decoder RNN

b)Decoder RNN Architecture : The first layer is an embedding layer having number of inputs as Vocab_size and outputs as ebedding size. The embedding layer basically converts eaxh word in a caption(vocabulary) to a vector which is then fed to the LSTM cells Layer along with the feature vector extracted from the Encoder CNN by concatenating both the vectors The Embeddings vector is created bt first removing the last token from the captions before concatenating with the feature vector I have used 3 LSTM layers stacked upon each other each having dropout parameter set to 0.4 to avoid over-fitting

Finally the Output of the Decoder RNN is calculated by passing the output of the LSTM layer as input to a fully connected linear layer

I selected the values of embed_size and hidden_size parameter by inferences drawn from going through the researcg papaers : 1) Show and Tell: A Neural Image Caption Generator (https://arxiv.org/pdf/1411.4555.pdf) 2) Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (https://arxiv.org/pdf/1502.03044.pdf)

I inferred that to prevent overfitting and getting more accurate results Deeper the Neural Net Model the better/more accurate would be the results Hence I chose Emded_size as 256 and Hidden_Size as 512 number of LSTM Cells Layes as 3 and Dropout value of LSTM layers as 0.4 BatchNormalization was also integrated in the Encoder CNN Model for getting better results and boosting the performance of the SEQ2SEQ model being built

I set the Vocab_threshold value to 5

I set the number of epochs to 4 for better more accurate results on completion of training

# RESULTS SHOWCASE
