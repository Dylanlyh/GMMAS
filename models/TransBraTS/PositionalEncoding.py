import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        # TODO: this is the implementation of learnable parameters using `nn.Parameter` method (convert a vanilla Tensor to a learnable one)
        # This is different from simply setting a tensor to be trainable (e.g. torch.tensor([1,2,3], requires_grad=True))

        self.position_embeddings = nn.Parameter(torch.zeros(1, 4096, 512)) #8x 原4096

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings
