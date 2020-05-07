from torch import nn
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (x, cond, label) in enumerate(loader):
        model.zero_grad()

        x = x.to(device)

        out, _ = model(x, condition=cond)

        loss = criterion(out, x)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == x).float()
        accuracy = correct.sum() / x.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
