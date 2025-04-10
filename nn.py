import torch
import torch.nn.functional as F


class Embedding:
    def __init__(self, num_embeddings, embedding_dim) -> None:
        self.emb = torch.randn(num_embeddings, embedding_dim)

    def __call__(self, idx):
        return self.emb[idx]

    def parameters(self):
        return [self.emb]


class Flatten:
    def __call__(self, x):
        if len(x.shape) == 2:
            return x
        B, T, E = x.shape
        return x.view(B, T * E)

    def parameters(self):
        return []


class Linear:
    def __init__(self, in_features, out_features):
        self.w = torch.randn(in_features, out_features) * 0.1
        self.b = torch.zeros(out_features)

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class Relu:
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []


class Sequential:
    def __init__(self, *args):
        self.layers = args

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Attention:
    def __init__(self, embed_dim, head_dim=None):
        self.head_dim = head_dim if head_dim else embed_dim
        self._query = torch.randn(embed_dim, self.head_dim)
        self._key = torch.randn(embed_dim, self.head_dim)
        self._value = torch.randn(embed_dim, self.head_dim)

    def __call__(self, x):
        query = x @ self._query
        key = x @ self._key
        value = x @ self._value

        weights = key @ query.permute(0, 2, 1)
        weights /= self.head_dim**0.5
        weights = F.softmax(weights, 2)

        out = weights @ value
        return out

    def parameters(self):
        return [self._query, self._key, self._value]
