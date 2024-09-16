import math

import torch
import typing

def create_model(
    cfg,
    input,
    subjects
):
    backends = torch.nn.ModuleDict({})
    for modality in cfg.modality:
        conf = getattr(cfg, modality)
        backend_type = conf.backend_type  # Make sure this is defined in your config

        if backend_type == "FFNN":
            backends[modality] = FFNN(
                input_size=input[modality].shape[0],
                hidden_size=conf.hidden_size,
                num_layers=conf.num_layers,
                dropout=conf.dropout,
                sigmoid=False,
                output_dim=conf.output_dim
            )
        elif backend_type == "NewFFNN":
            backends[modality] = NewFFNN(
                input_size=input[modality].shape[0],
                hidden_size=conf.hidden_size,
                num_layers=conf.num_layers,
                dropout=conf.dropout,
                sigmoid=False,
                output_dim=conf.output_dim
            )
        else:
            raise ValueError(f"Unknown backend_type: {backend_type}")

    if cfg.fusion == "concatenate":
        backend = ConcatenativeFusion(backends)
    elif cfg.fusion == "max":
        backend = MaxFusion(backends)
    elif cfg.fusion == "attention":
        backend = AttentionFusion(backends)
    elif cfg.fusion == "gated":
        backend = GatedFusion(backends)
    elif cfg.fusion == "m_attention":
        backend = MultiHeadAttention(backends)
    elif cfg.fusion == "s_attention":
        backend = AttentionBeforeConcatFusion(backends)
    elif cfg.fusion == "c_attention":
        backend = CrossAttentionFusion(backends)
    elif cfg.fusion == "mc_attention":
        backend = MultiHeadCrossModalAttention(backends)

    else:
        raise NotImplementedError(cfg.fusion)
    if cfg.frontend.name == "base":
        frontend = FFNN(
            input_size=backend.hidden_size,
            hidden_size=cfg.frontend.hidden_size,
            num_layers=cfg.frontend.num_layers,
            dropout=cfg.frontend.dropout,
            sigmoid=cfg.frontend.sigmoid,
            output_dim=len(cfg.targets)
        )
        model = BaseWrapper(backend, frontend)
    elif cfg.frontend.name == "submodels":
        frontend = PersonalizedFrontend(
            input_size=backend.hidden_size,
            hidden_size=cfg.frontend.hidden_size,
            num_layers=cfg.frontend.num_layers,
            dropout=cfg.frontend.dropout,
            sigmoid=cfg.frontend.sigmoid,
            output_dim=len(cfg.targets),
            subjects=subjects
        )
        model = PersonalizedWrapper(backend, frontend)
    elif cfg.frontend.name == "sub_attention":
        frontend = PersonalizedAttentionModel(
            input_size=backend.hidden_size,
            hidden_size=cfg.frontend.hidden_size,
            dropout=cfg.frontend.dropout,
            output_dim=len(cfg.targets),
            subjects=subjects
        )
        model = PersonalizedWrapper(backend, frontend)
    return model

class ConcatenativeFusion(torch.nn.Module):
    def __init__(
        self,
        backends
    ):
        super().__init__()
        self.backends = backends
        self.hidden_size = 0
        for key in self.backends.keys():
            self.hidden_size += self.backends[key].hidden_size
    
    def forward(self, x):
        return torch.cat([
            self.backends[key](x[key])
            for key in self.backends.keys()
        ], axis=1)

import torch
import math

class MultiHeadCrossModalAttention(torch.nn.Module):
    def __init__(self, backends, num_heads=2):
        super().__init__()
        self.backends = backends
        self.num_heads = num_heads

        # Assuming all backends have the same hidden_size
        self.hidden_size = self.backends['pcm'].hidden_size
        self.per_head_size = int(self.hidden_size) // int(self.num_heads)

        # Define weights for the query, key, value for both modalities
        self.query_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.out_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        # Get the output from both backends
        first_output = self.backends['pcm'](x['pcm'])
        second_output = self.backends['zcm'](x['zcm'])

        # Generate query, key, value for each modality
        query = self.query_weights(first_output).view(-1, self.num_heads, self.per_head_size)
        key = self.key_weights(second_output).view(-1, self.num_heads, self.per_head_size)
        value = self.value_weights(second_output).view(-1, self.num_heads, self.per_head_size)

        # Compute attention scores
        attention_weights = torch.nn.functional.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.per_head_size), dim=-1)

        # Compute attended values
        output = attention_weights @ value
        output = output.view(-1, self.hidden_size)

        # Apply output transformation
        return self.out_weights(output)


class MaxFusion(torch.nn.Module):
    def __init__(self, backends):
        super().__init__()
        self.backends = backends
        # Assign hidden_size attribute
        self.hidden_size = next(iter(backends.values())).hidden_size

    def forward(self, x):
        return torch.stack([self.backends[key](x[key]) for key in self.backends.keys()]).max(dim=0)[0]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, backends, num_heads=2):
        super().__init__()
        self.backends = backends
        self.num_heads = num_heads
        self.hidden_size = sum(backend.hidden_size for backend in backends.values())
        self.per_head_size = self.hidden_size // self.num_heads
        self.query_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        concatenated = torch.cat([self.backends[key](x[key]) for key in self.backends.keys()], dim=1)
        batch_size = concatenated.size(0)

        query = self.query_weights(concatenated).view(batch_size, self.num_heads, self.per_head_size)
        key = self.key_weights(concatenated).view(batch_size, self.num_heads, self.per_head_size)
        value = self.value_weights(concatenated).view(batch_size, self.num_heads, self.per_head_size)
        attention_weights = torch.nn.functional.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.per_head_size),
                                                        dim=-1)
        output = attention_weights @ value
        return output.view(batch_size, -1)


class AttentionFusion(torch.nn.Module):
    def __init__(self, backends):
        super().__init__()
        self.backends = backends
        total_hidden_size = sum(backend.hidden_size for backend in backends.values())
        self.hidden_size = total_hidden_size # Assign hidden_size attribute
        self.attention_weights = torch.nn.Sequential(
            torch.nn.Linear(total_hidden_size, total_hidden_size),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        concatenated = torch.cat([self.backends[key](x[key]) for key in self.backends.keys()], dim=1)
        attention = self.attention_weights(concatenated)
        return concatenated * attention


class GatedFusion(torch.nn.Module):
    def __init__(self, backends):
        super().__init__()
        self.backends = backends
        total_hidden_size = sum(backend.hidden_size for backend in backends.values())
        # Assign hidden_size attribute
        self.hidden_size = total_hidden_size
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(total_hidden_size, total_hidden_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        concatenated = torch.cat([self.backends[key](x[key]) for key in self.backends.keys()], dim=1)
        gates = self.gate(concatenated)
        return concatenated * gates


class SingleHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_weights = torch.nn.Linear(hidden_size, hidden_size)
        self.key_weights = torch.nn.Linear(hidden_size, hidden_size)
        self.value_weights = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        query = self.query_weights(x)
        key = self.key_weights(x)
        value = self.value_weights(x)
        attention_weights = torch.nn.functional.softmax(query @ key.transpose(-2, -1), dim=-1)
        return attention_weights @ value

class AttentionBeforeConcatFusion(torch.nn.Module):
    def __init__(self, backends):
        super().__init__()
        self.backends = backends
        self.attentions = {key: SingleHeadAttention(backend.hidden_size) for key, backend in backends.items()}
        self.hidden_size = sum(backend.hidden_size for backend in backends.values())

    def forward(self, x):
        attended_outputs = [self.attentions[key](self.backends[key](x[key])) for key in self.backends.keys()]
        return torch.cat(attended_outputs, dim=1)



class CrossAttentionFusion(torch.nn.Module):
    def __init__(self, backends):
        super().__init__()
        self.backends = backends

        # Assuming hidden_size attribute is present in the backends
        self.hidden_size = sum(backend.hidden_size for backend in backends.values())

        # Cross-attention between the first two backends
        self.cross_attention = SingleHeadAttention(backends["pcm"].hidden_size + backends["zcm"].hidden_size)

    def forward(self, x):
        # Apply the first two backends
        first_output = self.backends['zcm'](x['zcm'])
        second_output = self.backends['pcm'](x['pcm'])

        # Concatenate and apply cross-attention
        concatenated_outputs = torch.cat([first_output, second_output], dim=1)
        attended_output = self.cross_attention(concatenated_outputs)

        # Assuming no other backends

        return attended_output


class FFNN(torch.nn.Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 2,
        sigmoid: bool = False,
        dropout: float = 0.5
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sigmoid = sigmoid
        self.dropout = dropout

        layers = []
        layer_input = input_size
        for _ in range(num_layers - 1):
            layers += [
                torch.nn.Linear(layer_input, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            layer_input = hidden_size
        layers.append(torch.nn.Linear(layer_input, output_dim))
        if self.sigmoid:
            layers.append(torch.nn.Sigmoid())
        super().__init__(*layers)


class NewFFNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 4,
        sigmoid: bool = False,
        dropout: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sigmoid = sigmoid
        self.dropout = dropout

        layers = []
        layer_input = input_size

        # First layer with twice the hidden size
        layers += [
            torch.nn.Linear(layer_input, 2 * hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        layer_input = 2 * hidden_size

        # Loop to add remaining hidden layers, gradually decreasing the size
        for i in range(1, num_layers - 1):
            next_hidden_size = int(hidden_size * (num_layers - i) / (num_layers - 1))
            layers += [
                torch.nn.Linear(layer_input, next_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            layer_input = next_hidden_size

        # Final layer to output dimension
        layers.append(torch.nn.Linear(layer_input, output_dim))

        if self.sigmoid:
            layers.append(torch.nn.Sigmoid())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class BaseWrapper(torch.nn.Module):
    def __init__(self, backbone, frontend):
        super().__init__()
        self.backbone = backbone
        self.frontend = frontend

    def forward(self, x):
        return self.frontend(self.backbone(x))


class PersonalizedWrapper(BaseWrapper):
    def forward(self, x):
        return self.frontend({
            "features": self.backbone(x),
            "subject": x["subject"]
        })


class PersonalizedFrontend(torch.nn.Module):
    def __init__(
        self, 
        input_size: int,
        output_dim: int,
        subjects: typing.List[int],
        *, 
        num_layers: int = 4,
        sigmoid: bool = True,
        dropout: int = 0.5,
        hidden_size=30,
    ):
        super().__init__()
        assert num_layers % 2 == 0

        self.backend = FFNN(
            input_size=input_size,
            output_dim=hidden_size,
            dropout=dropout,
            sigmoid=False,
            num_layers=num_layers//2,
            hidden_size=hidden_size
        )
        self.frontends = torch.nn.ModuleDict({
            str(int(subject)): FFNN(
                input_size=hidden_size,
                output_dim=output_dim,
                dropout=dropout,
                sigmoid=False,
                num_layers=num_layers//2,
                hidden_size=hidden_size
            )
            for subject in subjects
        })

    def forward(self, x):
        embeddings = self.backend(x["features"])
        outputs = []
        for index, subject in enumerate(x["subject"].data.cpu()):
            outputs.append(self.frontends[str(int(subject.numpy()))](embeddings[index, :].unsqueeze(0)).squeeze(0))
        return torch.stack(outputs, dim=0)

class SimpleFFNNWithAttention(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_heads = num_heads

        # First linear layer
        self.linear1 = torch.nn.Linear(input_size, hidden_size)

        # Multi-head attention layer
        self.multi_head_attention = torch.nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        # Second linear layer
        self.linear2 = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2) # Required to match the expected input shape for multi-head attention
        attn_output, _ = self.multi_head_attention(x, x, x)
        x = attn_output.permute(1, 0, 2) # Restore the original shape
        x = attn_output.squeeze(0)
        x = self.linear2(x)
        return x

class PersonalizedAttentionModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_dim: int,
        subjects: typing.List[int],
        hidden_size: int = 30,
        num_heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()

        # Common backend processing with the new attention model
        self.backend = SimpleFFNNWithAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            output_dim=hidden_size, # Note that the output size is hidden_size to match the frontend input
            num_heads=num_heads,
            dropout=dropout
        )

        # Personalized frontends for each subject using the same attention model
        self.frontends = torch.nn.ModuleDict({
            str(int(subject)): SimpleFFNNWithAttention(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for subject in subjects
        })

    def forward(self, x):
        embeddings = self.backend(x["features"])
        outputs = []
        for index, subject in enumerate(x["subject"].data.cpu()):
            outputs.append(self.frontends[str(int(subject.numpy()))](embeddings[index, :].unsqueeze(0)).squeeze(0))
        return torch.stack(outputs, dim=0)
