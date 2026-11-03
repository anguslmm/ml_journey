from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt

type Loader = data.DataLoader[tuple[torch.Tensor, ...]]

block = 8 # block size, cannot change because of the current architecture

# hyper parameters defaults
default_batch = 64 # batch size
default_n_emb = 24 # embedding dimension
default_n_hidden = 200 # hidden layer size
default_epochs = 5
default_seed = 42

def load_data(filename: str = "./names.txt") -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocab: set[str] = set()
    names: list[str] = []
    with open(filename) as f:
        names = [n.strip()+'.' for n in f.readlines()]
    vocab = set(''.join(names))
    vocab.add('.')
    stoi = {s:i for i,s in enumerate(sorted(vocab))}
    itos = {i:s for i,s in enumerate(sorted(vocab))}
    return names, stoi, itos


def create_dataloaders(
    names: list[str],
    stoi: dict[str, int],
    seed: int = default_seed,
    batch: int = default_batch,
    splits: tuple[int, int, int] = (80, 10, 10),
) -> tuple[Loader, Loader, Loader]:
    g = torch.Generator()
    g.manual_seed(seed)
    indices: list[int] = torch.randperm(len(names), generator=g).tolist()
    shuffled = [names[i] for i in indices]

    def build_dataset(ns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        X: list[list[int]] = []
        Y: list[int] = []
        for n in ns:
            context = [0] * block
            for c in n:
                ix = stoi[c]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)

    training, dev, val = splits
    total = training + dev + val
    n1 = int(len(shuffled)*training/total)
    n2 = int(len(shuffled)*(total-val)/total)

    Xtr, Ytr = build_dataset(shuffled[:n1])
    Xdev, Ydev = build_dataset(shuffled[n1:n2])
    Xval, Yval = build_dataset(shuffled[n2:])

    train_set = data.DataLoader(data.TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True, drop_last=True, num_workers=0, generator=g)
    dev_set = data.DataLoader(data.TensorDataset(Xdev, Ydev), batch_size=batch, shuffle=False, drop_last=False, num_workers=0, generator=None)
    val_set = data.DataLoader(data.TensorDataset(Xval, Yval), batch_size=batch, shuffle=False, drop_last=False, num_workers=0, generator=None)
    return train_set, dev_set, val_set

def create_model(vocab_size: int, n_emb: int = default_n_emb, n_hidden: int = default_n_hidden, dropout_rate: float = 0.0, activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    model = nn.Sequential(
        nn.Embedding(vocab_size, n_emb),
        
        nn.Unflatten(1, (block//2, 2)),
        nn.Flatten(start_dim=2),
        nn.Linear(n_emb * 2, n_hidden, bias=False),
        nn.LayerNorm(n_hidden), 
        activation(),
        nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
        
        nn.Unflatten(1, (block//4, 2)),
        nn.Flatten(start_dim=2),
        nn.Linear(n_hidden * 2, n_hidden, bias=False),
        nn.LayerNorm(n_hidden),
        activation(),
        nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),
        
        nn.Unflatten(1, (block//8, 2)),
        nn.Flatten(start_dim=2),
        nn.Linear(n_hidden * 2, n_hidden, bias=False),
        nn.LayerNorm(n_hidden),
        activation(),
        nn.Dropout(dropout_rate) if dropout_rate else nn.Identity(),

        nn.Flatten(),
        nn.Linear(n_hidden, vocab_size),
    )
    return model


def create_optimizer(
    model: nn.Module,
    opt_name: Literal["SGD", "Adam"],
    sched_name: Literal["MultiStep", "WarmupCosine"] | None = None,
    opt_kwargs: dict[str, Any] = {},
    sched_args: dict[str, Any] = {},
    param_groups: list[dict[str, Any]] | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    schedulers = torch.optim.lr_scheduler
    opt: torch.optim.Optimizer | None = None
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
    params = model.parameters() if param_groups is None else param_groups
    if opt_name == "SGD":
        opt = torch.optim.SGD(params, **opt_kwargs)
    if opt_name == "Adam":
        opt = torch.optim.AdamW(params, **opt_kwargs)
    assert opt is not None

    if sched_name == "MultiStep":
        total_steps = sched_args['total_steps']
        assert sched_args is not None, "Need total steps for scheduler"
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(total_steps*0.50), int(total_steps*0.75)], gamma=0.1)
    if sched_name == "WarmupCosine":
        total_steps = sched_args['total_steps']
        assert sched_args is not None, "Need total steps for scheduler"
        peak: float = sched_args['lr'] or 0.01
        warm: int = sched_args['warmup_steps'] or 0
        lin = schedulers.LinearLR(opt, start_factor=0.1, total_iters=warm)    
        cos = schedulers.CosineAnnealingLR(opt, T_max=total_steps-warm, eta_min=peak*0.1)    
        sched = schedulers.SequentialLR(opt, schedulers=[lin, cos], milestones=[warm])

    return opt, sched

def train_model(
    model: nn.Module,
    data: Loader,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int = default_epochs,
) -> list[float]:
    model.train()

    lossi: list[float] = []
    first = True
    Xb: torch.Tensor
    Yb: torch.Tensor
    for _ in range(epochs):
        for Xb, Yb in data:
            logits: torch.Tensor = model(Xb)
            loss = F.cross_entropy(logits, Yb)
            if first:
                first = False
                print("Starting loss:", loss)
            lossi.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched:
                sched.step()
    return lossi

def plot_loss(l: list[float]) -> None:
    plt.plot(torch.tensor(l[:len(l)-len(l)%100]).view(-1, 10).mean(1))

@torch.no_grad()
def evaluate_model(model: nn.Module, datasets: list[Loader] = []) -> list[float]:
    model.eval()
    def evaluate(loader: Loader) -> float:
        total, count = 0.0, 0
        Xb: torch.Tensor
        Yb: torch.Tensor
        for Xb, Yb in loader:
            logits: torch.Tensor = model(Xb)
            total += F.cross_entropy(logits, Yb, reduction='sum').item()
            count += Yb.shape[0]
        return total / count
    return [evaluate(d) for d in datasets]

@torch.no_grad()
def make_names(model: nn.Module, itos: dict[int, str], n: int, temperature: float = 1) -> list[str]:
  model.eval()
  names: list[str] = []
  for _ in range(n):
    context = [0] * block
    out = ""
    while True:
      x = torch.tensor([context])
      logits: torch.Tensor = model(x)
      probs = F.softmax(logits / temperature, dim=-1)
      context = context[1:] + [cast(int, torch.multinomial(probs, 1).item())]
      if context[-1] != 0:
        out += itos[context[-1]]
      else:
        break
    names.append(out)
  return names


