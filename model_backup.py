import torch
import typing
from tran import MyAttention

def create_model(
    cfg,
    input,
    subjects
):
    backends = torch.nn.ModuleDict({})
    for modality in cfg.modality:
        conf = getattr(cfg, modality)
        backends[modality] = FFNN(
            input_size=input[modality].shape[0],
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            dropout=conf.dropout,
            sigmoid=False,
            output_dim=conf.output_dim
        )
    if cfg.fusion == "concatenate":
        backend = ConcatenativeFusion(backends)
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
        # self.myattention = MyAttention(input_features=256, num_heads=4)
        # self.myattention = MyAttention(input_features=256, num_heads=4)
        # self.myattention = MyAttention(input_features=256, num_heads=4)

    def forward(self, x):
        # x = self.myattention(x)
        # x = self.myattention(x)
        # x = self.myattention(x)
        return torch.cat([
            self.backends[key](x[key])
            for key in self.backends.keys()
        ], axis=1)


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