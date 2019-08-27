import torch
import pandas as pd
from string import punctuation
from torch.nn import Linear
from torch.nn import Sigmoid, LogSoftmax
from torch.optim import SGD
from torch.nn import BCELoss

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Perceptron(torch.nn.Module):
    def __init__(self, input_shape, bias=True):
        super(Perceptron, self).__init__()
        self.linear = Linear(input_shape, 1, bias=True)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)

        return x


class MulticlassPerceptron(torch.nn.Module):
    def __init__(self, input_shape, output_shape, bias=True):
        super(MulticlassPerceptron, self).__init__()
        self.linear = Linear(input_shape, output_shape, bias=True)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)

        return x

#could be inside of Perceptron class, but should be modified a little bit.
def train(model, train_data, optim, criterion, epochs=10, test_data=None):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for it, example in enumerate(train_data):
            optim.zero_grad()
            f, t = example
            X = torch.FloatTensor(f)
            y = torch.FloatTensor(t)
            output = model.forward(X)
            loss = criterion(output.view(-1), y)
            total_loss += loss.data.numpy()
            loss.backward()

            optim.step()

        if test_data:
            model.eval()
            y_pred = []
            y_true = []
            threshold = 0.5
            for f, t in test_data:
                X = torch.FloatTensor(f)
                y = torch.FloatTensor(t)
                output = model.forward(X)
                y_true.append(y.data.numpy()[0])
                y_pred.append(output.data.numpy()[0])

            y_pred = [int(p >= threshold) for p in y_pred]
            a = accuracy_score(y_true, y_pred)

        total_loss /= (it + 1)
        print("Epoch Loss: {:.2f}, Validation Accuracy: {:.2f}".format(total_loss, a))

    return model
#
# max_features = 1000
#
# model = Perceptron(max_features)
# criterion = BCELoss()
# optim = SGD(params=model.parameters(), lr=0.01)
#
#
# data = pd.read_pickle('./data/1a_acl_imdb.pkl')
#
#
# def clean_text(text):
#     return ''.join([c for c in text.lower() if c not in punctuation])
#
#
# data['cleaned'] = data['text'].map(clean_text)
#
# tfidf = TfidfVectorizer()
# tfidf.fit(data['cleaned'])
#
# max_features = 1000
# tfidf = TfidfVectorizer(max_features=max_features)
#
# features = tfidf.fit_transform(data['cleaned']).todense()
# labels = data.label.values.reshape(-1, 1)
#
# all_data = list(zip(features, labels))
# train_data, test_data = train_test_split(all_data, stratify=labels, random_state=42)
#
# train(model, train_data, optim, criterion, test_data=test_data)