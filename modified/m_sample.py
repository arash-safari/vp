import sys

sys.path.append('../')

import torch
from torchvision.utils import save_image
from tqdm import tqdm


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def make_sample(model_vqvae, model_top, model_middle, model_bottom, _dir, filename, batch=16, device='cuda', temp=1.0):
    top_sample = sample_model(model_top, device, batch, [32, 32], temp)

    if model_middle is not None:
        middle_sample = sample_model(
            model_middle, device, batch, [64, 64], temp, condition=top_sample
        )
        bottom_sample = sample_model(
            model_bottom, device, batch, [128, 128], temp, condition=middle_sample
        )
    else:
        bottom_sample = sample_model(
            model_bottom, device, batch, [64, 64], temp, condition=top_sample
        )

    if model_middle is not None:
        decoded_sample = model_vqvae.decode_code(top_sample, middle_sample, bottom_sample)
    else:
        decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)

    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, _dir + filename,
               normalize=True, range=(-1, 1))
