import os
import click
import torch
from timm.optim import Lamb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, RandomCrop, Compose, Resize
from torchvision.transforms.functional import to_pil_image

from models.networks import StyleTranfer
from dataset.dataset import ImageDataset, Transform, ConfigDict
from tqdm import tqdm

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        style_dataloader, 
        content_dataloader, 
        batch_size, 
        dest,
        total_nimg, 
        device="cuda",
    ):
        self.device = device
        self.output_dir = self.get_outdir(dest)
        os.makedirs(self.output_dir)
        self.writer = SummaryWriter(self.output_dir)

        self.model = model.to(device)
        self.optimizer = optimizer
        self.style_dataloader = style_dataloader
        self.content_dataloader = content_dataloader
        self.batch_size = batch_size
        self.total_nimg = total_nimg
        self.snapshot_kimg = 10

    def fit(self):
        grid_style = next(self.style_dataloader).to(self.device)
        grid_content = next(self.content_dataloader).to(self.device)

        cur_nimg = 0
        cur_tick = 0
        self.model.train()
        print("Launching training...")
        with tqdm(initial=0, total=int(self.total_nimg)) as pbar:
            while cur_nimg < self.total_nimg:
                losses = self.train_step()
                cur_nimg += self.batch_size
                cur_tick += 1
                self.save_losses(losses, cur_nimg)

                if cur_tick % (self.snapshot_kimg * 1000) == 0:
                    self.save_grid(grid_style, grid_content, cur_nimg)
                    self.save_snapshot(cur_nimg)
                pbar.update(self.batch_size)

    def save_snapshot(self, cur_nimg):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        output_path = os.path.join(self.output_dir, f"snapshot-{str(cur_nimg // 1000).zfill(8)}.pth")
        torch.save(checkpoint, output_path)

    @torch.no_grad()
    def save_grid(self, grid_style, grid_content, cur_nimg):
        grid_mixed = self.model(grid_style, grid_content)
        grid = self.make_image_grid(grid_style, grid_content, grid_mixed)
        to_pil_image(grid).save(os.path.join(self.output_dir, f"img_{str(cur_nimg // 1000).zfill(8)}.png"))

    def train_step(self):
        style_image = next(self.style_dataloader).to(self.device)
        content_image = next(self.content_dataloader).to(self.device)

        _, target_embedding, mixed_image_embedding, style_activations, mixed_activations = self.model(style_image, content_image, is_training=True)

        content_loss = self.model.content_loss_fn(target_embedding, mixed_image_embedding)
        style_loss = self.model.style_loss_fn(mixed_activations, style_activations)
        loss = content_loss + style_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = ConfigDict(
            loss=loss.item(),
            style_loss=style_loss.item(),
            content_loss=content_loss.item(),
        )
        return result

    def save_losses(self, losses, cur_nimg):
        self.writer.add_scalar("Train/Loss/Style", losses.style_loss, cur_nimg)
        self.writer.add_scalar("Train/Loss/Content", losses.content_loss, cur_nimg)
        self.writer.add_scalar("Train/Loss/Total", losses.loss, cur_nimg)

    @staticmethod
    def make_image_grid(style_image, content_image, mixed_image):
        grid_style = make_grid(style_image.cpu())
        grid_content = make_grid(content_image.cpu())
        grid_mix = make_grid(torch.clip(mixed_image.cpu(), 0, 1))
        grid = torch.concat((grid_style, grid_content, grid_mix), dim=1)
        return grid

    @staticmethod
    def get_outdir(dest):
        os.makedirs(dest, exist_ok=True)
        num = len(os.listdir(dest))
        output_dir = os.path.join(dest, f"{str(num).zfill(4)}-adain")
        return output_dir


@click.command()
@click.option("--seed",              type=int, default=0)
@click.option("--source-content")
@click.option("--source-style")
@click.option("--dest",              type=str)
@click.option("--batch-size",        type=int)
def main(seed, source_content, source_style, dest, batch_size):
    torch.manual_seed(seed)

    transform = Transform(Compose([ToTensor(), Resize(512), RandomCrop(256)]))
    style_dataset = ImageDataset(source_style, transform)
    content_dataset = ImageDataset(source_content, transform)

    style_dataloader = cycle(DataLoader(style_dataset, batch_size, shuffle=True, drop_last=True))
    content_dataloader = cycle(DataLoader(content_dataset, batch_size, shuffle=True, drop_last=True))
    model = StyleTranfer()
    optimizer = Lamb(model.decoder.parameters(), lr=0.005)

    trainer = Trainer(model, optimizer, style_dataloader, content_dataloader, batch_size, dest, total_nimg=int(500e3))
    trainer.fit()


if __name__ == "__main__":
    main()
