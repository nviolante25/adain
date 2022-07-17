import os
from pathlib import Path
from dataset import is_image_ext
from shutil import copyfile
from PIL import Image
import numpy as np
from tqdm import tqdm
import click


click.command()
click.option("--source", type=str)
click.option("--dest", type=str)


def main(source, dest):
    os.makedirs(dest, exist_ok=True)

    num_image = 0
    paths = [
        str(f) for f in Path(source).rglob("*") if is_image_ext(f) and os.path.isfile(f)
    ]
    for path in tqdm(paths):
        try:
            image = np.array(Image.open(path))
        except:
            continue
        if (len(image.shape) == 3) and (image.shape[2] == 3):
            image_path = os.path.join(dest, f"img{str(num_image).zfill(8)}.png")
            copyfile(path, image_path)
            num_image += 1


if __name__ == "__main__":
    main()
