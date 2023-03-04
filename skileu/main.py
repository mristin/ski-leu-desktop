"""Ski through the forest."""

import argparse
import importlib
import os
import pathlib
import sys
from typing import List, Tuple, Optional

import cv2
import pygame
import pygame.freetype
from icontract import require, ensure, DBC


import skileu

assert skileu.__doc__ == __doc__

PACKAGE_DIR = (
    pathlib.Path(str(importlib.resources.files(__package__)))
    if __package__ is not None
    else pathlib.Path(os.path.realpath(__file__)).parent
)

class Media:
    """Represent all the media loaded in the main memory from the file system."""

    def __init__(
        self,
        skier_sprite: pygame.surface.Surface,
        obstacle_sprites: List[pygame.surface.Surface],
        margin_sprites: List[pygame.surface.Surface],
        font: pygame.freetype.Font,  # type: ignore
    ) -> None:
        """Initialize with the given values."""
        self.skier_sprite = skier_sprite
        self.obstacle_sprites = obstacle_sprites
        self.margin_sprites = margin_sprites
        self.font = font


SCENE_WIDTH = 1024
SCENE_HEIGHT = 758

ROAD_MARGIN = 128


@ensure(lambda result: (result[0] is not None) ^ (result[1] is not None))
def load_media() -> Tuple[Optional[Media], Optional[str]]:
    """Load the media from the file system."""
    images_dir = PACKAGE_DIR / "media/images"

    pth = images_dir / "skier.png"
    try:
        skier_sprite = pygame.image.load(str(pth)).convert_alpha()
    except Exception as exception:
        return None, f"Failed to load {pth}: {exception}"

    pths = sorted(images_dir.glob("obstacle-*.png"))
    if len(pths) == 0:
        return None, f"No 'obstacle-*.png' found in image directory {images_dir}"
    obstacle_sprites = []  # type: List[pygame.surface.Surface]
    for pth in pths:
        try:
            sprite = pygame.image.load(str(pth)).convert_alpha()
        except Exception as exception:
            return None, f"Failed to load {pth}: {exception}"

        obstacle_sprites.append(sprite)

    pths = sorted(images_dir.glob("margin-*.png"))
    if len(pths) == 0:
        return None, f"No 'margin-*.png' found in image directory {images_dir}"
    margin_sprites = []  # type: List[pygame.surface.Surface]
    for pth in pths:
        try:
            sprite = pygame.image.load(str(pth)).convert_alpha()
        except Exception as exception:
            return None, f"Failed to load {pth}: {exception}"

        margin_sprites.append(sprite)

    pth = PACKAGE_DIR / "media/fonts/freesansbold.ttf"
    try:
        font = pygame.freetype.Font(str(pth))  # type: ignore
    except Exception as exception:
        return None, f"Failed to load {pth}: {exception}"

    return Media(
        skier_sprite=skier_sprite,
        obstacle_sprites=obstacle_sprites,
        margin_sprites=margin_sprites,
        font=font
    ), None

# TODO (mristin, 2023-03-4): continue from here

def main() -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()



    return 0


if __name__ == "__main__":
    sys.exit(main())
