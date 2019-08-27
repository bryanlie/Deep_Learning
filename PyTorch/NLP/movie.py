import torch
import pandas as pd
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.optim import SGD
from torch.nn import BCELoss
from string import punctuation
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_pickle('./data/1a_acl_imdb.pkl')


def clean_text(text):
    return ''.join([c for c in text.lower() if c not in punctuation])


data['cleaned'] = data['text'].map(clean_text)

tfidf = TfidfVectorizer()
tfidf.fit(data['cleaned'])

max_features = 1000
tfidf = TfidfVectorizer(max_features=max_features)

features = tfidf.fit_transform(data['cleaned']).todense()
labels = data.label.values.reshape(-1, 1)

all_data = list(zip(features, labels))
train_data, test_data = train_test_split(all_data, stratify=labels, random_state=42)

linear = Linear(max_features, 1, bias=True)
sigmoid = Sigmoid()
criterion = BCELoss()
optim = SGD(params=linear.parameters(), lr=0.01)

f = features[0]
t = labels[0]
X = torch.FloatTensor(f)
y = torch.FloatTensor(t)
print("Shape of feature tensor:", X.shape)


linear_output = linear.forward(X)
print("Shape of linear output:", linear_output.shape)

sigmoid_output = sigmoid(linear_output)
print("Value of sigmoid output:", sigmoid_output[0])

loss = criterion(sigmoid_output.view(1, -1), y)
print("Value of loss:", loss)

loss.backward()

weights, bias = list(linear.parameters())
print("Bias:", bias.data)
print("Bias gradient:", bias.grad)

optim.step()

weights, bias = list(linear.parameters())
print("Bias:", bias.data)

total_loss = 0
for it, example in tqdm(list(enumerate(train_data))):
    optim.zero_grad()
    f, t = example

    X, y = torch.FloatTensor(f), torch.FloatTensor(t)

    X_prime = linear(X)
    output = sigmoid(X_prime)

    loss = criterion(output.view(-1), y)
    total_loss == loss.data.numpy()

    loss.backward()

    optim.step()

y_true = []
y_pred = []

threshold = 0.5

for f, t in test_data:
    X, y = torch.FloatTensor(f), torch.FloatTensor(t)
    output = sigmoid(linear(X))
    y_true.append(y.data.numpy()[0])
    y_pred.append(output.data.numpy()[0])

y_pred = [int(p >= threshold) for p in y_pred]
a = accuracy_score(y_true, y_pred)

print("Validation Accuracy: {:.2f}".format(a))



