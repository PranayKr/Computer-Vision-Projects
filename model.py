import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        #features = self.embed(features)
        features = self.bn(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,dropout=0.4)
        self.linear = nn.Linear(hidden_size, vocab_size)
        #pass
    
    def forward(self, features, captions):
        features = features.view(len(features), 1, -1)
        embeddings = self.embed(captions[:, :-1])
        #embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        inputs = torch.cat((features, embeddings), 1)
        ltsm_out, hidden = self.lstm(inputs)
        outputs = self.linear(ltsm_out)
        #hiddens, _ = self.lstm(embeddings)
        #outputs = self.linear(hiddens[0])       
        return outputs
        #pass

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids
        #pass