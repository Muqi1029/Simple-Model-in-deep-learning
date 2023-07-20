from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class WordWindowClassifier(nn.Module):
    def __init__(self, hyperparameters, vocab, pad_ix=0):
        super(WordWindowClassifier, self).__init__()
        self.window_size = hyperparameters['window_size']
        self.hidden_dim = hyperparameters['hidden_dim']
        self.embedded_dim = hyperparameters['embedded_dim']

        self.embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.embedded_dim, padding_idx=pad_ix)

        full_window_size = 2 * self.window_size + 1
        self.model = nn.Sequential(
            nn.Linear(in_features=full_window_size * self.embedded_dim, out_features=self.hidden_dim),
            nn.Linear(in_features=self.hidden_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        :param inputs: a (B, L) tensor of token indices
        :return:
        """
        inputs = inputs.unfold(dimension=1, size=self.window_size * 2 + 1, step=1)
        B, L, _ = inputs.shape
        x = self.embed(inputs)
        x = x.view(B, L, -1)  # B * L * (window_size * embedded_dim)
        output = self.model(x)
        output = output.view(B, -1)
        return output


# Processing
def custom_collate_fn(batch, window_size, word2idx):
    x, y = zip(*batch)

    # 1. pad the sentence
    def pad_window(sentence, window_size, pad_token='<pad>'):
        pad = window_size * [pad_token]
        return pad + sentence + pad

    x = [pad_window(sentence, window_size) for sentence in x]

    # 2.
    def convert_tokens_to_index(x, word2idx):
        return [[word2idx.get(word, word2idx['<unk>']) for word in sentence] for sentence in x]

    x = convert_tokens_to_index(x, word2idx=word2idx)

    # 3. pad the number array
    pad_token_idx = word2idx['<pad>']
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_idx)

    lengths = [len(label) for label in y]
    lengths = torch.LongTensor(lengths)
    y = [torch.LongTensor(y_i) for y_i in y]
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return x_padded, y_padded, lengths


# Train
def train(data, labels, model_hyperparameters, num_epoch=1000):
    """

    :param data: ['I am in China', 'Are you in America now?']
    :param labels: [[0, 0, 0, 1], [0, 0, 0, 1, 0]]
    :param model_hyperparameters: contains['window_size', 'hidden_dim', 'embedded_dim']
    :param num_epoch: the number of epoch
    :return:
    """

    # precess: construct vocabulary list
    data = [sentence.lower().split() for sentence in data]
    vocab = set(word for sentence in data for word in sentence)
    vocab.add('<pad>')
    vocab.add('<unk>')
    idx2word = sorted(list(vocab))
    vocab_size = len(idx2word)
    word2idx = dict(zip(idx2word, range(vocab_size)))

    # loader
    collate_fn = partial(custom_collate_fn, window_size=model_hyperparameters['window_size'], word2idx=word2idx)
    loader = DataLoader(list(zip(data, labels)), batch_size=model_hyperparameters['batch_size'], shuffle=True,
                        collate_fn=collate_fn)

    # model
    model = WordWindowClassifier(model_hyperparameters, idx2word)

    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss
    def loss_function(batch_outputs, batch_labels, batch_lengths):
        bceloss = nn.BCELoss()
        loss = bceloss(batch_outputs, batch_labels.float())
        return loss / batch_lengths.sum().float()

    # train
    for _ in tqdm(range(num_epoch)):
        total_loss = 0
        for batch_inputs, batch_labels, batch_lengths in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_labels, batch_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if _ % 100 == 0:
            print(f"loss: {total_loss}")

    return model, word2idx