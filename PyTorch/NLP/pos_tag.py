import torch
import pandas as pd
from torch.optim import SGD
from torch.nn import NLLLoss
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_pickle('./data/3_penn_treebank_pos.pkl')


all_words = set(itertools.chain.from_iterable(data['text']))
all_labels = set(itertools.chain.from_iterable(data['label']))

word2idx = {word: idx for idx, word in enumerate(all_words)}
idx2word = {idx: word for word, idx in word2idx.items()}

label2idx = {word: idx for idx, word in enumerate(all_labels)}
idx2label = {idx: word for word, idx in label2idx.items()}

vocab_size = len(all_words)
label_size = len(all_labels)

features = data['text'].map(lambda x: [word2idx[i] for i in x]).tolist()
labels = data['label'].map(lambda x: [label2idx[i] for i in x]).tolist()

train_data, test_data = train_test_split(list(zip(features, labels)))

from tagging import POSTagger

model = POSTagger(vocab_size=vocab_size, embedding_dim=100, hidden_dim=50, output_dim=label_size, batch_size=1)
optim = SGD(params=model.parameters(), lr=0.01)
criterion = NLLLoss()

for i in range(10):
    total_loss = 0
    model.train()
    y_true_train = []
    y_pred_train = []
    for it, example in enumerate(train_data):
        f, t = example
        X = torch.LongTensor(f)
        y = torch.LongTensor(t)

        model.hidden = model.init_hidden()
        output = model.forward(X).squeeze(1)
        optim.zero_grad()
        prediction = torch.argmax(output, dim=1)
        loss = criterion(output, y)
        total_loss += loss.data.numpy()
        y_true_train.extend(list(y.data.numpy()))
        y_pred_train.extend(list(prediction.numpy()))
        loss.backward()

        optim.step()

    model.eval()
    y_pred = []
    y_true = []
    for example in test_data:
        optim.zero_grad()
        f, t = example
        X = torch.LongTensor(f)
        y = torch.LongTensor(t)

        model.hidden = model.init_hidden()
        output = model.forward(X).squeeze(1)
        prediction = torch.argmax(output, dim=1)

        y_true.extend(list(y.data.numpy()))
        y_pred.extend(list(prediction.numpy()))

    a = accuracy_score(y_true, y_pred)
    a_train = accuracy_score(y_true_train, y_pred_train)
    total_loss /= (it + 1)

    print("Loss: {:.2f}, Training Accuracy: {:.2f}, Validation Accuracy: {:.2f}".format(total_loss, a_train, a))


# embedding = Embedding(num_embeddings=vocab_size, embedding_dim=100)
#
# lstm = LSTM(input_size=100, hidden_size=50)
#
# h0 = torch.zeros(1, 1, 50)
# c0 = torch.zeros(1, 1, 50)
# lstm_hidden = h0, c0
#
# linear = Linear(50, label_size)
#
# softmax = LogSoftmax(dim=2)
#
# criterion = NLLLoss()
#
# f = features[0]
# t = labels[0]
# X = torch.LongTensor(f)
# y = torch.LongTensor(t)
# print("Integer Feature Sequence Shape:", X.shape)
# print("Integer Target Shape:", y.shape)
#
# embedded_sequence = embedding(X)
# print("Embedding Sequence Shape:", embedded_sequence.shape)
#
# embedded_sequence = embedded_sequence.unsqueeze(1)
#
# lstm_output, lstm_hidden = lstm(embedded_sequence, lstm_hidden)
# print("LSTM Output Shape:", lstm_output.shape)
#
# final_output = lstm_output
# print("Linear Layer Input Shape:", final_output.shape)
#
# linear_output = linear(final_output)
# print("Linear Output Shape:", linear_output.shape)
#
# softmax_output = softmax(linear_output)
# print("Softmax Output Shape:", softmax_output.shape)
#
# softmax_squeezed = softmax_output.squeeze(1)
# print("Softmax Squeezed Shape:", softmax_squeezed.shape)
# print("Target Shape:", y.shape)
#
# loss = criterion(softmax_squeezed, y)
# print("Loss Value:", loss.data.numpy())