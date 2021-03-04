import torch
import torch.nn as nn
import numpy as np

class ExhaustiveModel(nn.Module):

    def __init__(self, hidden_size, n_tags, max_region, embedding_url=None, bidirectional=True, lstm_layers=1,
                 n_embeddings=None, embedding_dim=None, freeze=False, char_feat_dim=100, n_chars = 100):
        super().__init__()

        if embedding_url:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(np.load(embedding_url)),
                freeze=freeze
            )
        else:
            self.embedding = nn.Embedding(n_embeddings, embedding_dim, padding_idx=0)

        self.embedding_dim = self.embedding.embedding_dim
        self.char_feat_dim = char_feat_dim
        self.word_repr_dim = self.embedding_dim + self.char_feat_dim

        self.char_repr = CharLSTM(
            n_chars=n_chars,
            embedding_size=char_feat_dim // 2,
            hidden_size=char_feat_dim // 2,
        ) if char_feat_dim > 0 else None

        self.dropout = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(
            input_size=self.word_repr_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.lstm_layers = lstm_layers
        self.n_tags = n_tags
        self.max_region = max_region
        self.n_hidden = (1 + bidirectional) * hidden_size

        self.region_clf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3 * self.n_hidden, n_tags),
            # nn.Softmax(),
        )

    def forward(self, sentences, sentence_lengths, sentence_words, sentence_word_lengths,
                sentence_word_indices):
        word_repr = self.embedding(sentences)
        
        if self.char_feat_dim > 0:
            char_feat = self.char_repr(sentence_words, sentence_word_lengths, sentence_word_indices)
            word_repr = torch.cat([word_repr, char_feat], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
        out, (hn, _) = self.lstm(packed)

        max_sent_len = sentences.shape[1]
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=max_sent_len, batch_first=True)
        unpacked = unpacked.transpose(0, 1)

        max_len = sentence_lengths[0]
        regions = list()
        for region_size in range(1, self.max_region + 1):
            for start in range(0, max_len - region_size + 1):
                end = start + region_size
                regions.append(torch.cat([unpacked[start], torch.mean(unpacked[start:end], dim=0),
                                          unpacked[end - 1]], dim=-1))
                                          
        output = torch.stack([self.region_clf(region) for region in regions], dim=-1)
        return output


class CharLSTM(nn.Module):

    def __init__(self, n_chars, embedding_size, hidden_size, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.n_hidden = hidden_size * (1 + bidirectional)

        self.embedding = nn.Embedding(n_chars, embedding_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True,
        )

    def sent_forward(self, words, lengths, indices):
        sent_len = words.shape[0]
        
        embedded = self.embedding(words)
        
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        _, (hn, _) = self.lstm(packed)
        
        hn = hn.permute(1, 0, 2).contiguous().view(sent_len, -1)
        
        hn[indices] = hn
        return hn

    def forward(self, sentence_words, sentence_word_lengths, sentence_word_indices):

        batch_size = len(sentence_words)
        batch_char_feat = torch.nn.utils.rnn.pad_sequence(
            [self.sent_forward(sentence_words[i], sentence_word_lengths[i], sentence_word_indices[i])
             for i in range(batch_size)], batch_first=True)

        return batch_char_feat