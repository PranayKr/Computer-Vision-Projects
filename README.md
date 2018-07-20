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

![humanactivity_1](https://user-images.githubusercontent.com/25223180/43006613-c703cf3c-8c53-11e8-9852-ddc7755a91ce.PNG)

![catbird](https://user-images.githubusercontent.com/25223180/43006420-4aa15dce-8c53-11e8-815c-7b4af49397f6.PNG)

![groupofcows](https://user-images.githubusercontent.com/25223180/43006222-c50b3c98-8c52-11e8-9bd3-e93ee3269e0f.PNG)

![living room](https://user-images.githubusercontent.com/25223180/43006235-ce3f4b4c-8c52-11e8-9263-2df410ba044d.PNG)

![giraffe](https://user-images.githubusercontent.com/25223180/43006247-d5f69ad4-8c52-11e8-8326-a79fa92a730e.PNG)

![airport](https://user-images.githubusercontent.com/25223180/43006260-de8e8ac6-8c52-11e8-9362-06301f434b84.PNG)

![banana](https://user-images.githubusercontent.com/25223180/43006279-e926806a-8c52-11e8-811f-7efbb8887180.PNG)

![bench](https://user-images.githubusercontent.com/25223180/43006295-f4645966-8c52-11e8-88f4-5aa472344cd6.PNG)

![bicycle](https://user-images.githubusercontent.com/25223180/43006310-fc37ccb8-8c52-11e8-8baa-b1f1e9f793eb.PNG)

![bird_beach](https://user-images.githubusercontent.com/25223180/43006320-03e2cc74-8c53-11e8-995c-c20401c8a01f.PNG)

![bird1](https://user-images.githubusercontent.com/25223180/43006329-0b9fa5f4-8c53-11e8-8bd7-b0feea11b9bf.PNG)

![boat_lake](https://user-images.githubusercontent.com/25223180/43006339-13f664cc-8c53-11e8-90f1-1e594f6de0dc.PNG)

![boat2](https://user-images.githubusercontent.com/25223180/43006350-1c08fc9c-8c53-11e8-8e0f-5e2072e350bd.PNG)

![bus](https://user-images.githubusercontent.com/25223180/43006364-2495d3ee-8c53-11e8-9d68-3982c57c4415.PNG)

![bus1](https://user-images.githubusercontent.com/25223180/43006378-2b9824d0-8c53-11e8-8139-5fa3ced265cb.PNG)

![cat](https://user-images.githubusercontent.com/25223180/43006394-353ee12c-8c53-11e8-8679-2ff21158d587.PNG)

![cat_laptop](https://user-images.githubusercontent.com/25223180/43006402-3bf89364-8c53-11e8-8307-c7d06470c4b6.PNG)

![cat3](https://user-images.githubusercontent.com/25223180/43006410-43932d46-8c53-11e8-81ee-a0f1b07a023b.PNG)

![clock](https://user-images.githubusercontent.com/25223180/43006449-6106441c-8c53-11e8-8aaa-f996de89c689.PNG)

![clocktower](https://user-images.githubusercontent.com/25223180/43006463-6bdd07c2-8c53-11e8-9f2a-53702d967f2a.PNG)

![donut](https://user-images.githubusercontent.com/25223180/43006483-775e144c-8c53-11e8-8fc7-6349446dd9d6.PNG)

![donuts](https://user-images.githubusercontent.com/25223180/43006492-7e1d1a44-8c53-11e8-85df-97cbaa477cf0.PNG)

![doubledeckerbus](https://user-images.githubusercontent.com/25223180/43006516-8684f076-8c53-11e8-85a9-851a07c7240c.PNG)

![elephant](https://user-images.githubusercontent.com/25223180/43006533-8df3fb5e-8c53-11e8-9999-e595fc8df4b4.PNG)

![fire_hydrant](https://user-images.githubusercontent.com/25223180/43006547-9612bfe6-8c53-11e8-8ee9-90adf39d3d5f.PNG)

![firehydrant1](https://user-images.githubusercontent.com/25223180/43006563-9f7a4432-8c53-11e8-930d-7599564ef067.PNG)

![foodplate](https://user-images.githubusercontent.com/25223180/43006572-a8126570-8c53-11e8-9aab-d3bef1cfb8a4.PNG)

![fridge](https://user-images.githubusercontent.com/25223180/43006585-b11044d0-8c53-11e8-8c04-95bf668ac56c.PNG)

![hotdog](https://user-images.githubusercontent.com/25223180/43006606-c04b1326-8c53-11e8-911f-1f0e2b6168e9.PNG)

![humanactivity_2](https://user-images.githubusercontent.com/25223180/43006652-e0e3201a-8c53-11e8-92f5-f48d47b69946.PNG)

![humanactivity_3](https://user-images.githubusercontent.com/25223180/43006668-eb7f0cf0-8c53-11e8-9390-6d0c0223797d.PNG)

![humanactivity_4](https://user-images.githubusercontent.com/25223180/43006681-f53453fe-8c53-11e8-92b0-5075cbcff222.PNG)

![humanactivity_5](https://user-images.githubusercontent.com/25223180/43006693-fd7b0f6c-8c53-11e8-836c-2eb0d598a2dc.PNG)

![humanactivity_6](https://user-images.githubusercontent.com/25223180/43006712-0654f2ce-8c54-11e8-9a69-fef5eea8b7a3.PNG)

![humanactivity_7](https://user-images.githubusercontent.com/25223180/43006722-0eea14fa-8c54-11e8-8651-0eb5b5ec218f.PNG)

