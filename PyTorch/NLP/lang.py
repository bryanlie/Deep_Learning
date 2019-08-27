import torch
import pandas as pd
from torch.nn import Linear
from torch.nn import Sigmoid, LogSoftmax
from torch.optim import SGD
from torch.nn import BCELoss, NLLLoss
from string import punctuation
import itertools
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

data = pd.read_pickle('./data/1b_stackoverflow_qna.pkl')
print(data.head())


def clean_text(text):
    return ''.join([c for c in text.lower() if c not in punctuation])


data['cleaned'] = data['text'].map(clean_text)

tfidf = TfidfVectorizer()
tfidf.fit(data['cleaned'])
print("Total tokens in input corpus: {}".format(len(tfidf.vocabulary_)))

max_features = 1000
tfidf = TfidfVectorizer(max_features=max_features)

features = tfidf.fit_transform(data['cleaned']).todense()

le = LabelEncoder()
labels = le.fit_transform(data.label.values).reshape(-1, 1)
label_size = len(le.classes_)

all_data = list(zip(features, labels))
train_data, test_data = train_test_split(all_data, stratify=labels, random_state=42)

linear = Linear(max_features, label_size, bias=True)
softmax = LogSoftmax(dim=1)
criterion = NLLLoss()
optim = SGD(params=linear.parameters(), lr=0.01)

f = features[0]
t = labels[0]
X = torch.FloatTensor(f)
y = torch.LongTensor(t)
print("Shape of feature tensor:", X.shape)
print("Shape of label tensor:", y.shape)

linear_output = linear(X)
print("Shape of linear output:", linear_output.shape)

softmax_output = softmax(linear_output)
print("Shape of softmax output:", softmax_output.shape)

softmax_norm = torch.exp(softmax_output).sum()
print("Softmax normalization:", softmax_norm)

loss = criterion(softmax_output, y)
print("Loss:", loss.data)

from perceptron import MulticlassPerceptron

model = MulticlassPerceptron(max_features, label_size)
optim = SGD(params=model.parameters(), lr=0.01)
criterion = NLLLoss()

LEARNING_RATE = 0.01
EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    linear.train()
    total_loss = 0
    for it, example in list(enumerate(train_data)):
        optim.zero_grad()
        f, t = example
        X = torch.FloatTensor(f)
        y = torch.LongTensor(t)
        output = model.forward(X)
        loss = criterion(output, y)
        total_loss += loss.data.numpy()
        loss.backward()

        optim.step()

    model.eval()
    y_pred = []
    y_true = []
    threshold = 0.5

    for f, t in test_data:
        X = torch.FloatTensor(f)
        y = torch.FloatTensor([t])
        output = model.forward(X)
        y_true.append(y.data.numpy()[0])
        y_pred.append(torch.argmax(output.data).numpy())

    a = accuracy_score(y_true, y_pred)

    total_loss /= (it + 1)

    print("Loss: {:.2f}, Validation Accuracy: {:.2f}".format(total_loss, a))

