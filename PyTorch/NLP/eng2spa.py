import torch
import numpy as np
import pandas as pd
import pickle
from torch.nn import Linear, Embedding, RNN, GRU, LSTM
from torch.nn import Sigmoid, LogSoftmax
from torch.optim import SGD, Adam
from torch.nn import BCELoss, NLLLoss, CrossEntropyLoss
from string import punctuation
import itertools
from tqdm import tqdm
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

data = pd.read_pickle('./data/4_europarl_en_sp.pkl')
data['text'] = data['english'].map(lambda x: "".join([i for i in x.lower() if i not in string.punctuation]).split())
data['label'] = data['spanish'].map(lambda x: "".join([i for i in x.lower() if i not in string.punctuation]).split())

data['text'] = data['text'].map(lambda x: ['<SOS>'] + x + ['<EOS>'])
data['label'] = data['label'].map(lambda x: ['<SOS>'] + x + ['<EOS>'])
print(data.shape)
print(data.head())


input_words = set(itertools.chain.from_iterable(data['text']))
output_words = set(itertools.chain.from_iterable(data['label']))

input2idx = {word: idx for idx, word in enumerate(input_words)}
idx2input = {idx: word for word, idx in input2idx.items()}

output2idx = {word: idx for idx, word in enumerate(output_words)}
idx2output = {idx: word for word, idx in output2idx.items()}

input_size = len(input_words)
output_size = len(output_words)

input_seqs = data['text'].map(lambda x: [input2idx[i] for i in x]).tolist()
output_seqs = data['label'].map(lambda x: [output2idx[i] for i in x]).tolist()

data = list(zip(input_seqs, output_seqs))

train_data, test_data = train_test_split(data)


enc_vocab_size = input_size
enc_embedding_dim = 100
enc_hidden_dim = 50

dec_vocab_size = output_size
dec_embedding_dim = 100
dec_hidden_dim = 50
dec_output_dim = output_size

from autocoder import Encoder, Decoder, Seq2seq

enc = Encoder(enc_vocab_size, enc_embedding_dim, enc_hidden_dim, batch_size=1)
dec = Decoder(dec_vocab_size, dec_embedding_dim, dec_hidden_dim, dec_output_dim, batch_size=1)
s2s = Seq2seq(enc, dec)


optim = SGD(params=s2s.parameters(), lr=0.01)
criterion = NLLLoss()

epochs = 1
for epoch in range(epochs):
    s2s.train()
    total_loss = 0
    s2s.train()
    for it, example in enumerate(train_data):

        if (it % 100 == 0) and (it != 0):
            print("Epoch|it: {}|{}, Total Loss: {:.2f}".format(epoch, it, total_loss / it))
        input_seq, output_seq = example
        optim.zero_grad()

        input_seq = torch.LongTensor(input_seq)
        output_seq = torch.LongTensor(output_seq)

        res = s2s.forward(input_seq, output_seq[:-1], p_tf=0.5)
        loss = criterion(res, output_seq[1:])
        loss.backward()
        total_loss += loss.data.numpy()

        optim.step()
#
# input_seq = input_seqs[0]
# output_seq = output_seqs[0]
#
# sos_idx = output2idx['<SOS>']
# eos_idx = output2idx['<EOS>']
# pred_idxs = s2s.predict(input_seq, sos_idx, eos_idx)
# print([idx2output[i] for i in pred_idxs])

# enc_embedding = Embedding(num_embeddings=input_size, embedding_dim=100)
# enc_rnn = LSTM(input_size=100, hidden_size=50)
# dec_embedding = Embedding(num_embeddings=output_size, embedding_dim=100)
# dec_rnn = LSTM(input_size=100, hidden_size=50)
# dec_linear = Linear(50, output_size)
# softmax = LogSoftmax(dim=1)
# criterion = NLLLoss()
#
# input_seq = input_seqs[0]
# output_seq = output_seqs[0]
#
# input_seq_tensor = torch.LongTensor(input_seq)
# output_seq_tensor = torch.LongTensor(output_seq)
# print("Input Sequence Tensor Shape:", input_seq_tensor.shape)
# print("Output Sequence Tensor Shape:", output_seq_tensor.shape)
#
# enc_embedded = enc_embedding.forward(input_seq_tensor)
# print("Encoder Embedded Sequence Shape:", enc_embedded.shape)
#
# enc_embedded_unsqueezed = enc_embedded.unsqueeze(1)
# print("Encoder Embedded Sequence Shape (1 batch):", enc_embedded.shape)
#
# h0 = torch.zeros(1, 1, 50)
# c0 = torch.zeros(1, 1, 50)
# enc_hidden = (h0, c0)
#
# enc_out, enc_hidden = enc_rnn(enc_embedded_unsqueezed, enc_hidden)
# print("Encoder Output Shape:", enc_out.shape)
# print("Encoder Hidden Shape(s):", enc_hidden[0].shape, enc_hidden[1].shape)
#
#
# dec_hidden = enc_hidden
#
# dec_embedded = dec_embedding(output_seq_tensor)
# print("Decoder Embedded Sequence Shape:", dec_embedded.shape)
#
# dec_embedded_unsqueezed = dec_embedded.unsqueeze(1)
# print("Decoder Embedded Sequence Shape (1 batch):", dec_embedded.shape)
#
#
# dec_lstm_in = dec_embedded_unsqueezed[:-1]
# print("Decoder LSTM Input Shape:", dec_lstm_in.shape)
#
# dec_out, dec_hidden = dec_rnn(dec_lstm_in, dec_hidden)
# print("Decoder Output Shape:", dec_out.shape)
# print("Decoder Hidden Shape(s):", dec_hidden[0].shape, dec_hidden[1].shape)
#
# dec_linear_output = dec_linear(dec_out).squeeze(1)
# print("Decoder Linear Output Shape:", dec_linear_output.shape)
#
# dec_softmax_output = softmax(dec_linear_output)
# print("Decoder Softmax Output Shape:", dec_softmax_output.shape)
#
# dec_softmax_norms = torch.exp(dec_softmax_output).sum(dim=1)
# print("Decoder Softmax Norms Shape:", dec_softmax_norms.shape)
# print("Decoder Softmax Norms:", dec_softmax_norms)
#
#
# dec_loss_target = output_seq_tensor[1:]
# print("Decoder Loss Target Shape:", dec_loss_target.shape)
#
#
# loss = criterion(dec_softmax_output, dec_loss_target)
# print("Loss:", loss.data)

