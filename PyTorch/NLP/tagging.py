import torch
from torch.nn import Linear, Embedding, LSTM
from torch.nn import LogSoftmax


class POSTagger(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, batch_size):
        super(POSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.linear = Linear(hidden_dim, output_dim)
        self.batch_size = batch_size
        self.softmax = LogSoftmax(dim=2)
        self.hidden = self.init_hidden()

    def forward(self, x):
        e = self.embedding(x)
        e = e.view(len(x), self.batch_size, -1)
        out, self.hidden = self.rnn(e, self.hidden)
        output = self.linear(out)
        so = self.softmax(output).squeeze(1)
        return so

    def init_hidden(self):
        h0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

