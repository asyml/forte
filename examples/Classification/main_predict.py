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
    wordlistlist = [] # Store all sentences, each sentence is a list

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
imdb_test_reader = IMDBReader(cache_in_memory=True)

traininput = transform_reader(imdb_train_reader)
trainlabellist = traininput[0]
trainwordlistlist = traininput[1]


testinput = transform_reader(imdb_test_reader)
testlabellist = testinput[0] # Do not use label in test
testwordlistlist = testinput[1]


orderedlabel=list(set(trainlabellist.copy()))
orderedlabel.sort() # Let label to a fix number each time
label2num = {}
num2label = {}
i = 0

for label in orderedlabel:
    label2num[label] = i
    num2label[i] = label
    i = i + 1

class MyDataset(data.Dataset):
    def __init__(self, xlistlist):
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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        return X

test_dataset = MyDataset(testwordlistlist)
test_loader_args = dict(shuffle = False, batch_size=1,
                         num_workers=8, pin_memory = True) \
    if cuda else dict(shuffle=False, batch_size=1)
test_loader = data.DataLoader(test_dataset, **test_loader_args)

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
model.load_state_dict(torch.load("1epochembedsl.t7") )
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
#print(model)

def pred_model(model, test_loader):
    with torch.no_grad():
        model.eval()
        predLabel = []

        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            outputs = model(data)
            value, predicted = torch.max(outputs.data, 1)

            predLabel = predLabel + predicted.tolist()

    return predLabel

predLabel =pred_model(model,test_loader)

for pred in predLabel:
    print(num2label[pred])