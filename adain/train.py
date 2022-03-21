from torch.optim import Adam
import torch
import torch.nn as nn
from torch import Generator
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, RandomCrop, Compose, Normalize, Resize
from itertools import cycle

from networks import StyleTranfer
from networks import style_loss_fn

from tqdm import tqdm

if __name__ == "__main__":
    # Dataset
    style_dataset = ImageFolder(
        "/home/nviolante/datasets/style_transfer",
        transform=Compose(
            [Resize(512), RandomCrop(256), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        ),
    )
    content_dataset = ImageFolder(
        "/home/nviolante/datasets/coco",
        transform=Compose(
            [Resize(512), RandomCrop(256), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        ),
    )
    batch_size = 8
    style_dataloader = DataLoader(style_dataset, batch_size=batch_size, num_workers=4, drop_last=True)
    content_dataloader = DataLoader(content_dataset, batch_size=batch_size, num_workers=4, drop_last=True)

    max_epochs = 2
    content_loss_fn = nn.MSELoss()
    net = StyleTranfer()
    optimizer = Adam(net.decoder.parameters())

    for epoch in tqdm(range(max_epochs)):
        # Training phase
        for (style_image, content_image) in tqdm(zip(cycle(style_dataloader), content_dataloader)):
            optimizer.zero_grad()

            mixed_embedding, mixed_image_embedding, style_activations, mixed_activations = net(
                style_image[0], content_image[0], is_training=True
            )
            content_loss = content_loss_fn(mixed_embedding, mixed_image_embedding)
            style_loss = style_loss_fn(mixed_activations, style_activations)
            loss = content_loss + style_loss

            loss.backward()
            optimizer.step()

    print()
