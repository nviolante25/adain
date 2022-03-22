import os
from tqdm import tqdm
from itertools import cycle
from PIL import ImageFile
import torch
import torch.nn as nn
from torch.multiprocessing import set_sharing_strategy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, RandomCrop, Compose, Resize

from models.networks import StyleTranfer
from models.networks import style_loss_fn


def create_log_grid(style_image, content_image, mixed_image):
    grid_style = make_grid(style_image.cpu())
    grid_content = make_grid(content_image.cpu())
    grid_mix = make_grid(torch.clip(mixed_image.cpu(), 0, 1))
    grid_log = torch.concat((grid_style, grid_content, grid_mix), dim=1)
    return grid_log


def split(dataset, percentage=0.95):
    train_size = int(percentage * len(dataset))
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    return train_dataset, val_dataset


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    set_sharing_strategy("file_system")

    # Dataset
    style_dataset = ImageFolder(
        "/home/nviolante/datasets/style_transfer",
        transform=Compose([Resize(512), RandomCrop(256), ToTensor()]),
    )
    content_dataset = ImageFolder(
        "/home/nviolante/datasets/coco",
        transform=Compose([Resize(512), RandomCrop(256), ToTensor()]),
    )

    device = "cuda"
    max_epochs = 100
    batch_size = 4
    tensorboard_grid_save_freq = 200  # num_steps before logging results
    model_save_freq = 10  # epochs before saving model

    # For overfit use Subset(style_dataset_train, [5917]) and Subset(content_dataset_train, [5])
    style_dataset_train, style_dataset_val = split(style_dataset)
    style_dataloaders = {
        "train": DataLoader(style_dataset_train, batch_size=batch_size, drop_last=True, shuffle=True),
        "val": DataLoader(style_dataset_val, batch_size=batch_size, drop_last=True, shuffle=True),
    }
    content_dataset_train, content_dataset_val = split(content_dataset)
    content_dataloaders = {
        "train": DataLoader(content_dataset_train, batch_size=batch_size, drop_last=True, shuffle=True),
        "val": DataLoader(content_dataset_val, batch_size=batch_size, drop_last=True, shuffle=True),
    }

    content_loss_fn = nn.MSELoss()
    model = StyleTranfer().to(device)
    optimizer = Adam(model.decoder.parameters(), lr=0.0005)
    writer = SummaryWriter()
    output_dir = "/home/nviolante/workspace/adain/results"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(max_epochs):
        # Training phase
        pbar = tqdm(
            enumerate(zip(cycle(style_dataloaders["train"]), content_dataloaders["train"])),
            total=len(content_dataset_train),
        )
        for num_step, (style_image, content_image) in pbar:
            optimizer.zero_grad()
            style_image = style_image[0].to(device)
            content_image = content_image[0].to(device)

            mixed_image, target_embedding, mixed_image_embedding, style_activations, mixed_activations = model(
                style_image, content_image, is_training=True
            )
            content_loss = content_loss_fn(target_embedding, mixed_image_embedding)
            style_loss = style_loss_fn(mixed_activations, style_activations)
            loss = content_loss + style_loss

            loss.backward()
            optimizer.step()

            # Progress bar updates
            pbar.set_description(f"Epoch [{epoch}/{max_epochs}]")
            pbar.set_postfix(content_loss=content_loss.item(), style_loss=style_loss.item())

            # Tensorboard logs
            writer.add_scalar("Train/Style loss", style_loss.item(), (num_step + 1) * (epoch + 1))
            writer.add_scalar("Train/Content loss", content_loss.item(), (num_step + 1) * (epoch + 1))
            writer.add_scalar("Train/Total loss", loss.item(), (num_step + 1) * (epoch + 1))

            if num_step % tensorboard_grid_save_freq == 0:
                grid_log = create_log_grid(style_image, content_image, mixed_image)
                writer.add_image("Train/ Step Style-Content-Mix Image", grid_log, (num_step + 1) * (epoch + 1))

        grid_log = create_log_grid(style_image, content_image, mixed_image)
        writer.add_image("Train/Style-Content-Mix Image", grid_log, epoch)

        # Evaluation phase
        with torch.no_grad():
            for num_step, (style_image, content_image) in enumerate(
                zip(cycle(style_dataloaders["val"]), content_dataloaders["val"])
            ):
                style_image = style_image[0].to(device)
                content_image = content_image[0].to(device)

                mixed_image, target_embedding, mixed_image_embedding, style_activations, mixed_activations = model(
                    style_image, content_image, is_training=True
                )
                content_loss = content_loss_fn(target_embedding, mixed_image_embedding)
                style_loss = style_loss_fn(mixed_activations, style_activations)
                loss = content_loss + style_loss

                writer.add_scalar("Val/Style loss", style_loss.item(), (num_step + 1) * (epoch + 1))
                writer.add_scalar("Val/Content loss", content_loss.item(), (num_step + 1) * (epoch + 1))
                writer.add_scalar("Val/Total loss", loss.item(), (num_step + 1) * (epoch + 1))

            grid_log = create_log_grid(style_image, content_image, mixed_image)
            writer.add_image("Val/Style-Content-Mix Image", grid_log, epoch)

        if epoch % model_save_freq == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_{epoch}.pt"))
