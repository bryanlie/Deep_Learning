import torch
import pandas as pd
from string import punctuation
import itertools


from torch.optim import SGD
from torch.nn import NLLLoss

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_pickle('./data/2_r8.pkl')

def clean_text(text):
    return ''.join([c for c in text.lower() if c not in punctuation])


data['cleaned'] = data['text'].map(clean_text)

text_split = data['cleaned'].map(lambda x: x.split())
all_words = set(list(itertools.chain.from_iterable(text_split)))
vocal_size = len(all_words)

all_labels = list(data['label'].unique())
label_size = len(all_labels)

word2idx = {word: idx for idx, word in enumerate(all_words)}
idx2word = {idx: word for word, idx in word2idx.items()}

label2idx = {label: idx for idx, label in enumerate(all_labels)}
idx2label = {idx: label for label, idx in label2idx.items()}

data['text_encoded'] = data['cleaned'].map(lambda x: [word2idx[word] for word in x.split()])
data['label_encoded'] = data['label'].map(lambda x: [label2idx[label] for label in x.split()])

labels = data['label_encoded'].values
features = data['text_encoded']
train_data, test_data = train_test_split(list(zip(features, labels)))

# embedding = Embedding(num_embeddings=vocal_size, embedding_dim=100)
# lstm = LSTM(input_size=100, hidden_size=50)
#
# h0 = torch.zeros(1, 1, 50)
# c0 = torch.zeros(1, 1, 50)
# lstm_hidden = h0, c0
#
# linear = Linear(50, label_size)
# softmax = LogSoftmax(dim=1)
#
# criterion = NLLLoss()
#
# f = features[0]
# t = labels[0]
# X = torch.LongTensor(f)
# y = torch.LongTensor(t)
# print("Shape of integer feature sequence: ", X.shape)
# print("Shape of integer target: ", y.shape)
#
# embedded_seq = embedding(X)
# print("Shape of embedding sequence: ", embedded_seq.shape)
#
# embedded_seq = embedded_seq.unsqueeze(1)
#
# lstm_output, lstm_hidden = lstm(embedded_seq, lstm_hidden)
# print("Shape of LSTM output: ", lstm_output.shape)
#
# final_output = lstm_output[-1]
# print("Shape of linear layer input: ",  final_output.shape)
#
# linear_output = linear(final_output)
# print("Shape of linear output: ", linear_output.shape)
#
# softmax_output = softmax(linear_output)
# print("Shape of softmax output: ", softmax_output.shape)
# print("Shape of target: ", y.shape)
#
# loss = criterion(softmax_output, y)
# print("Loss value: ", loss.data.numpy())

from rnn_clf import RNNClassifier

model = RNNClassifier(vocal_size=vocal_size, embedding_dim=100, hidden_dim=50, output_dim=label_size, batch_size=1)
optim = SGD(params=model.parameters(), lr=0.01)
criterion = NLLLoss()

for i in range(10):
    total_loss = 0
    model.train()
    for it, ex in enumerate(train_data):
        f, t = ex
        X = torch.LongTensor(f)
        y = torch.LongTensor(t)

        model.hidden = model.init_hidden()
        output = model.forward(X)
        optim.zero_grad()
        pred = torch.argmax(output)
        loss = criterion(output, y)
        total_loss += loss.data.numpy()

        loss.backward()

        optim.step()

    model.eval()
    y_pred = []
    y_true = []
    for ex in test_data:
        optim.zero_grad()
        f, t = ex
        X = torch.LongTensor(f)
        y = torch.LongTensor(t)
        model.hidden = model.init_hidden()
        output = model.forward(X)
        pred = torch.argmax(output)

        y_true.append(y.data.numpy()[0])
        y_pred.append(torch.argmax(output.data).numpy())

        a = accuracy_score(y_true, y_pred)

    total_loss /= (it + 1)

    print("Loss: {:.2f}, Validation Accuracy: {:.2f}".format(total_loss, a))

