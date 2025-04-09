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


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads) -> None:
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({num_heads})"
        self.attentions = [Attention(embed_dim, int(embed_dim / num_heads)) for _ in range(num_heads)]
        self.w0 = torch.randn(embed_dim, embed_dim) * 0.01

    def __call__(self, x):
        out = [att(x) for att in self.attentions]
        out = torch.concat(out, -1) @ self.w0
        return out

    def parameters(self):
        return [p for att in self.attentions for p in att.parameters()] + [self.w0]


class LayerNorm:
    def __call__(self, x):
        eps = 1e-10
        std = (x.var(-1, keepdim=True) + eps) ** 0.5
        mean = x.mean(-1, keepdim=True)
        return (x - mean) / std

    def parameters(self):
        return []


class Sublayer:
    def __init__(self, f) -> None:
        self.f = f
        self.layer_norm = LayerNorm()

    def __call__(self, x):
        return self.layer_norm(x + self.f(x))

    def parameters(self):
        return self.f.parameters() + self.layer_norm.parameters()


class FeedForward:
    def __init__(self, in_features, out_features, inner_features) -> None:
        self.net = Sequential(
            Linear(in_features, inner_features),
            Relu(),
            Linear(inner_features, out_features),
        )

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()


class EncodingBlock:
    def __init__(self, embed_dim, num_heads, inner_features) -> None:
        self.attentional_sublayer = Sublayer(MultiHeadAttention(embed_dim, num_heads))
        self.feedforward_sublayer = Sublayer(FeedForward(embed_dim, embed_dim, inner_features))
        self.sublayers = Sequential(self.attentional_sublayer, self.feedforward_sublayer)

    def __call__(self, x):
        return self.sublayers(x)

    def parameters(self):
        return self.sublayers.parameters()
