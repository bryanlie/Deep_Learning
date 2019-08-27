import torch
import numpy as np
from torch.nn import Linear, Embedding, LSTM
from torch.nn import LogSoftmax


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.batch_size = batch_size
        self.softmax = LogSoftmax()
        self.hidden = self.init_hidden()

    def forward(self, x):
        e = self.embedding(x)
        e = e.view(len(x), self.batch_size, -1)
        out, self.hidden = self.rnn(e, self.hidden)
        return out, self.hidden

    def init_hidden(self):
        h0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, batch_size):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.linear = Linear(hidden_dim, output_dim)
        self.batch_size = batch_size
        self.softmax = LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def forward(self, input, hidden):
        self.hidden = hidden
        e = self.embedding(input)
        e = e.view(len(input), self.batch_size, -1)
        out, self.hidden = self.rnn(e, self.hidden)
        self.out = out
        output = self.linear(out[0])
        so = self.softmax(output)
        return so, self.hidden

    def init_hidden(self):
        h0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)


class Seq2seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, input_seq, output_seq, p_tf=0):
        outputs = []

        self.enc.hidden = self.enc.init_hidden()
        self.dec.hidden = self.dec.init_hidden()

        enc_output, enc_hidden = self.enc.forward(torch.LongTensor(input_seq))
        dec_hidden = enc_hidden
        tf_cnt = 0
        for i in range(output_seq.shape[0]):

            if (np.random.uniform()) > p_tf and (i != 0):
                dec_input = torch.LongTensor([torch.argmax(dec_output).data])
            else:
                dec_input = torch.LongTensor([output_seq[i]])

            dec_output, dec_hidden = self.dec.forward(dec_input, dec_hidden)
            outputs.append(dec_output)

        return torch.stack(outputs).squeeze(1)

    def predict(self, input_seq, sos_idx, eos_idx, max_len=20):
        outputs = []
        self.enc.hidden = self.enc.init_hidden()
        self.dec.hidden = self.dec.init_hidden()

        enc_output, enc_hidden = self.enc.forward(torch.LongTensor(input_seq))
        dec_hidden = enc_hidden

        cnt = 0
        dec_input = torch.LongTensor([sos_idx])

        dec_output, dec_hidden = self.dec.forward(dec_input, dec_hidden)

        output_idx = torch.argmax(dec_output).data

        while (int(output_idx) != eos_idx) and (cnt <= max_len):
            cnt += 1
            dec_input = torch.LongTensor([output_idx])
            dec_output, dec_hidden = self.dec.forward(dec_input, dec_hidden)
            output_idx = torch.argmax(dec_output).data
            outputs.append(int(output_idx))

        return outputs

