from base64 import decode
from turtle import back, forward
import torch
import torch.nn as nn
from torchvision.models import vgg19

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: change zero padding to reflect padding
        backbone = vgg19(pretrained=True).features

        backbone = nn.Sequential(*[backbone[i] for i in range(21)])

        for param in backbone.parameters():
            param.requires_grad = False

        layers_style_loss = {"relu1_1": 1, "relu2_1": 6, "relu3_1": 11, "relu4_1": 20}
        for layer in layers_style_loss.values():
            backbone[layer].register_forward_hook(self._save_activations)

        self.encoder = backbone
        self.activations = []

    def forward(self, image):
        latent_code = self.encoder(image)
        activations = self.activations
        self.activations = []
        return latent_code, activations

    def _save_activations(self, model, inputs, outputs):
        self.activations.append(outputs)


class AdaIN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = vgg19(pretrained=True).features

    def forward(self, content, style):
        content_mean = torch.mean(content, dim=(-2, -1), keepdim=True)
        content_std = torch.std(content, dim=(-2, -1), keepdim=True)
        style_mean = torch.mean(style, dim=(-2, -1), keepdim=True)
        style_std = torch.std(style, dim=(-2, -1), keepdim=True)
        return style_std * (content - content_mean) / (content_std + 1e-8) + style_mean
        

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode="reflect"),
        )

    def forward(self, embedding):
        return self.decoder(embedding)


class StyleTranfer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.encoder.eval()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def forward(self, content_image, style_image, is_training=False):
        content_embedding, _ = self.encoder(content_image)
        style_embedding, style_activations = self.encoder(style_image)

        mixed_embedding = self.adain(content_embedding, style_embedding)
        mixed_image = self.decoder(mixed_embedding)
        mixed_image_embedding, mixed_activations = self.encoder(mixed_image)

        if is_training:
            return mixed_embedding, mixed_image_embedding, style_activations, mixed_activations
        return mixed_image

        
def style_loss_fn(mixed_activations, style_activations):
    assert len(style_activations) == len(mixed_activations)
    style_loss = 0
    num_layers = len(mixed_activations)
    for i in range(num_layers):
        style_loss += torch.norm(style_activations[0].mean(dim=(-2, -1)) - mixed_activations[0].mean(dim=(-2, -1)))
        style_loss += torch.norm(style_activations[0].std(dim=(-2, -1)) - mixed_activations[0].std(dim=(-2, -1)))
    return style_loss / num_layers



if __name__ == '__main__':
    content_loss_fn = nn.MSELoss()
    batch_size = 14
    image = torch.randn(size=(batch_size, 3, 256, 256))
    encoder = Encoder()
    adain = AdaIN()
    decoder = Decoder()
    embeddings, activations = encoder(image)
    style_activations = [activations[i][batch_size//2:] for i in range(4)]
    style_embedding = embeddings[batch_size//2:]
    content_embedding = embeddings[:batch_size//2]
    mixed_embedding = adain(content_embedding, style_embedding)

    mixed_image = decoder(mixed_embedding)
    encoded_mixed_image, mixed_activations = encoder(mixed_image)
    
    # loss computation
    content_loss = content_loss_fn(mixed_embedding, encoded_mixed_image)
    style_loss = style_loss_fn(mixed_activations, style_activations)
    

    print()

    max_epoch = 100