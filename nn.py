import torch


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
