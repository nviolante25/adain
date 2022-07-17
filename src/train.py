import os
import click
import torch
from timm.optim import Lamb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, RandomCrop, Compose, Resize

from models.networks import StyleTranfer
from dataset.dataset import ImageDataset, Transform, InfiniteSampler, ConfigDict


class Trainer:
    def __init__(
        self,
        dest,
        device="cuda",
    ):
        self.device = device
        self.output_dir = self.get_outdir(dest)
        os.makedirs(self.output_dir)
        self.writer = SummaryWriter(self.output_dir)

    def fit(
        self,
        model,
        optimizer,
        dataloaders,
        batch_size,
        snapshot_interval=1000,
        total_images=100e3,
        train_kwargs=None,
    ):
        model.to(self.device)
        self._state = ConfigDict(
            num_images=0,
            num_batches=0,
            tick=0,
            snapshot_interval=snapshot_interval,
            total_images=total_images,
        )

        grid_style = next(dataloaders.style).to(self.device)
        grid_content = next(dataloaders.content).to(self.device)

        done = False
        while not done:
            losses = self.training_step(model, optimizer, dataloaders)
            self.save_losses(losses)
            if self._time_to_save():
                self.save_snapshot(model, optimizer)
                self.save_grid(model, grid_style, grid_content)
                self.print_progress(losses)
                self._state.tick += 1
            self._state.num_images += batch_size
            self._state.num_batches += 1
            done = self._state.num_images >= total_images

    def _time_to_save(self):
        return (
            self._state.num_images - (self._state.tick * self._state.snapshot_interval)
            > 0
        )

    def print_progress(self, losses):
        tick = self._state.tick
        print(f"tick {tick}: style {losses.style_loss}, content {losses.content_loss}")

    def save_snapshot(self, model, optimizer):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": self._state,
        }
        tick = self._state.tick
        output_path = os.path.join(
            self.output_dir, f"snapshot-{str(tick).zfill(8)}.pth"
        )
        torch.save(checkpoint, output_path)

    @torch.no_grad()
    def save_grid(self, model, grid_style, grid_content):
        grid_mixed = model(grid_style, grid_content)
        grid = self.make_image_grid(grid_style, grid_content, grid_mixed)
        self.writer.add_image(
            "Train/ Step Style-Content-Mix Image", grid, self._state.num_images
        )

    def training_step(self, model, optimizer, dataloaders):
        optimizer.zero_grad()
        style_image = next(dataloaders.style).to(self.device)
        content_image = next(dataloaders.content).to(self.device)

        (
            _,
            target_embedding,
            mixed_image_embedding,
            style_activations,
            mixed_activations,
        ) = model(style_image, content_image, is_training=True)

        content_loss = model.content_loss_fn(target_embedding, mixed_image_embedding)
        style_loss = model.style_loss_fn(mixed_activations, style_activations)
        loss = content_loss + style_loss

        loss.backward()
        optimizer.step()

        result = ConfigDict(
            loss=loss.item(),
            style_loss=style_loss.item(),
            content_loss=content_loss.item(),
        )
        return result

    def save_losses(self, losses):
        step = self._state.num_images
        self.writer.add_scalar("Train/Loss/Style", losses.style_loss, step)
        self.writer.add_scalar("Train/Loss/Content", losses.content_loss, step)
        self.writer.add_scalar("Train/Loss/Total", losses.loss, step)

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
@click.option("--seed", type=int, default=0)
@click.option("--source-content")
@click.option("--source-style")
@click.option("--dest", type=str)
@click.option("--batch-size", type=int)
def main(seed, source_content, source_style, dest, batch_size):
    torch.manual_seed(seed)

    transform = Transform(Compose([ToTensor(), Resize(512), RandomCrop(256)]))
    style_dataset = ImageDataset(source_style, transform)
    content_dataset = ImageDataset(source_content, transform)

    style_dataloader = iter(
        DataLoader(
            style_dataset, batch_size, sampler=InfiniteSampler(style_dataset, seed)
        )
    )
    content_dataloader = iter(
        DataLoader(
            content_dataset, batch_size, sampler=InfiniteSampler(content_dataset, seed)
        )
    )

    dataloaders = ConfigDict(style=style_dataloader, content=content_dataloader)

    model = StyleTranfer()
    optimizer = Lamb(model.decoder.parameters(), lr=0.005)

    trainer = Trainer(dest)
    trainer.fit(model, optimizer, dataloaders, batch_size)


if __name__ == "__main__":
    main()
