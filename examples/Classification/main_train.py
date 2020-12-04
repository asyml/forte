#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn as nn
import torch
import random
import os
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from forte.data.readers.imdb_reader import IMDBReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence
from typing import Iterator, Iterable

cuda = torch.cuda.is_available()
print(cuda)

unkidx = 400000
padidx = 400001

word2vec = {}
index2word = {}
word2index = {}

def construct_word_embedding_table():
    index = 0
    # set up {word:vec, ...}
    with open("./glove.6B.300d.txt") as reader:
        for eachLine in reader:
            eachLineList = eachLine.strip(" ").split(" ")
            word = eachLineList[0]

            vec = eachLineList[1:]
            vec = [float(x) for x in vec]

            word2vec[word] = vec
            index2word[index] = word
            word2index[word] = index

            index = index + 1

    unkvec = word2vec["the"].copy()
    padvec = word2vec["apple"].copy()
    random.shuffle(unkvec)
    random.shuffle(padvec)

    l = []
    for index in range(0, len(word2vec)):
        if index % 100000 == 0:
            print(index)
        vector = word2vec[index2word[index]]
        l.append(vector)

    l.append(unkvec)
    l.append(padvec)
    embmatrix = torch.tensor(l)

    return embmatrix

def transform_reader(imdbreader):
    dataset_path: str = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        *([os.path.pardir] * 2),
        'data_samples/imdb'))

    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(imdbreader)
    pipeline.initialize()

    data_packs: Iterable[DataPack] = pipeline.process_dataset(dataset_path)
    labellist = []
    wordlistlist = [] #store all sentences, each sentence is a list

    # Each .imdb file is corresponding to an Iterable Obj
    for pack in data_packs:
        sentences: Iterator[Sentence] = pack.get(Sentence)

        for sentence in sentences:
            comment = sentence.text
            sentiment_label = sentence.sentiment[comment]

            wordlist = comment.strip(" ").rstrip().split(" ")

            labellist.append(sentiment_label)
            wordlistlist.append(wordlist)

    return (labellist, wordlistlist)


def generpad(num):
    pad = []
    for i in range(num):
        pad.append(padidx)
    return pad


imdb_train_reader = IMDBReader(cache_in_memory=True)
imdb_val_reader = IMDBReader(cache_in_memory=True)

traininput = transform_reader(imdb_train_reader)
trainlabellist = traininput[0]
trainwordlistlist = traininput[1]

valinput = transform_reader(imdb_val_reader)
validlabellist = valinput[0]
validwordlistlist = valinput[1]


orderedlabel=list(set(trainlabellist.copy()))
orderedlabel.sort() #let label to a fix number each time
label2num = {}
i = 0
for label in orderedlabel:
    label2num[label] = i
    i = i + 1

class MyDataset(data.Dataset):
    def __init__(self, xlistlist, ylist):
        sentencelist = []
        for xlist in xlistlist:
            indexlist = []
            for word in xlist:
                if word in word2vec:
                    wordidx = word2index[word]

                else:
                    wordidx = unkidx
                indexlist.append(wordidx)
            indexlist = indexlist + generpad(60 - len(indexlist))
            sentencelist.append(indexlist)
        self.x = torch.LongTensor(sentencelist)

        labellist = []
        for label in ylist:
            labellist.append(label2num[label])
        self.y = torch.LongTensor(labellist)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X = self.x[index]
        Y = self.y[index]

        return X, Y

train_dataset = MyDataset(trainwordlistlist,trainlabellist)
train_loader_args = dict(shuffle = True, batch_size=1,
                         num_workers=8, pin_memory = True) \
    if cuda else dict(shuffle=True, batch_size=1)
train_loader = data.DataLoader(train_dataset, **train_loader_args)


val_dataset = MyDataset(validwordlistlist,validlabellist)
val_loader_args = dict(shuffle = False, batch_size=1,
                       num_workers=8, pin_memory = True) \
    if cuda else dict(shuffle=False, batch_size=1)
val_loader = data.DataLoader(val_dataset,**val_loader_args)


class MyConv(nn.Module):
    def __init__(self, embmatrix):
        super(MyConv, self).__init__()
        self.embedding = nn.Embedding(*embmatrix.size())
        self.embedding.weight.data.copy_(embmatrix)
        self.embedding.weight.requires_grad = True

        self.conv1 = nn.Conv1d(300, 100, 3)
        self.conv2 = nn.Conv1d(300, 100, 4)
        self.conv3 = nn.Conv1d(300, 100, 5)
        self.pool1 = nn.MaxPool1d(58)
        self.pool2 = nn.MaxPool1d(57)
        self.pool3 = nn.MaxPool1d(56)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 16)

    def forward(self, x):  # for a sentence (300,60)
        # x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x)))
        x3 = self.pool3(F.relu(self.conv3(x)))
        x = torch.cat((x1, x2, x3), dim=1).squeeze()
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        return x

model = MyConv(construct_word_embedding_table())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
#print(model)

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        print("batch_idx", batch_idx)

        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        outputs = model(data)

        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss)
    return running_loss


def val_model(model, val_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            value, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
        running_loss /= len(val_loader)
        acc = (correct_predictions / total_predictions) * 100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc


n_epochs = 2

for i in range(n_epochs):
    print("Epoch: ", i+1)
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = val_model(model, val_loader, criterion)

    print('='*20)
    modelname = str(i+1) + "epochembedsl.t7"
    torch.save(model.state_dict(),modelname)

