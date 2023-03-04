"""Ski through the forest."""

import argparse
import enum
import importlib
import os
import pathlib
import random
import sys
from typing import List, Tuple, Optional, Union

import cv2
import pygame
import pygame.freetype
from icontract import require, ensure, DBC

import skileu
from skileu import common, bodypose

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
            skier_forward_sprite: pygame.surface.Surface,
            skier_left_sprite: pygame.surface.Surface,
            skier_right_sprite: pygame.surface.Surface,
            obstacle_sprites: List[pygame.surface.Surface],
            margin_sprites: List[pygame.surface.Surface],
            font: pygame.freetype.Font,  # type: ignore
    ) -> None:
        """Initialize with the given values."""
        self.skier_forward_sprite = skier_forward_sprite
        self.skier_left_sprite = skier_left_sprite
        self.skier_right_sprite = skier_right_sprite
        self.obstacle_sprites = obstacle_sprites
        self.margin_sprites = margin_sprites
        self.font = font


SCENE_WIDTH = 800
SCENE_HEIGHT = 600

ROAD_MARGIN = 128


@ensure(lambda result: (result[0] is not None) ^ (result[1] is not None))
def load_media() -> Tuple[Optional[Media], Optional[str]]:
    """Load the media from the file system."""
    images_dir = PACKAGE_DIR / "media/images"

    pth = images_dir / "skier-forward.png"
    try:
        skier_forward_sprite = pygame.image.load(str(pth)).convert_alpha()
    except Exception as exception:
        return None, f"Failed to load {pth}: {exception}"

    pth = images_dir / "skier-left.png"
    try:
        skier_left_sprite = pygame.image.load(str(pth)).convert_alpha()
    except Exception as exception:
        return None, f"Failed to load {pth}: {exception}"

    pth = images_dir / "skier-right.png"
    try:
        skier_right_sprite = pygame.image.load(str(pth)).convert_alpha()
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
        skier_forward_sprite=skier_forward_sprite,
        skier_left_sprite=skier_left_sprite,
        skier_right_sprite=skier_right_sprite,
        obstacle_sprites=obstacle_sprites,
        margin_sprites=margin_sprites,
        font=font
    ), None


class Obstacle:
    """Represent an obstacle in the level."""
    #: Appearance
    sprite: pygame.surface.Surface

    #: In world coordinates
    xy: Tuple[int, int]

    def __init__(
            self,
            sprite: pygame.surface.Surface,
            xy: Tuple[int, int]
    ) -> None:
        """Initialize with the given values."""
        self.sprite = sprite
        self.xy = xy


class Level:
    """Represent a single scene."""
    obstacles: List[Obstacle]

    def __init__(self, obstacles: List[Obstacle]) -> None:
        """Initialize with the given values."""
        self.obstacles = obstacles


def calculate_skier_width(media: Media) -> int:
    """Calculate the skier width based on the sprites."""
    return max(
        sprite.get_width()
        for sprite in [
            media.skier_forward_sprite,
            media.skier_left_sprite,
            media.skier_right_sprite
        ]
    )


def calculate_skier_height(media: Media) -> int:
    """Calculate the skier width based on the sprites."""
    return max(
        sprite.get_height()
        for sprite in [
            media.skier_forward_sprite,
            media.skier_left_sprite,
            media.skier_right_sprite
        ]
    )


def generate_level(media: Media) -> Level:
    """Generate randomly a level."""
    obstacles = []  # type: List[Obstacle]

    # region Generate left margin
    cursor = 0
    i = 0
    while cursor < SCENE_HEIGHT:
        sprite = media.margin_sprites[i % len(media.margin_sprites)]
        obstacles.append(
            Obstacle(
                sprite=sprite,
                xy=(0, cursor)
            )
        )
        cursor += sprite.get_height()
        i += 1
    # endregion

    margin_sprites_mirrored = [
        pygame.transform.flip(sprite, True, False)
        for sprite in media.margin_sprites
    ]

    # region Generate right margin
    cursor = 0
    i = 0
    while cursor < SCENE_HEIGHT:
        sprite = margin_sprites_mirrored[i % len(margin_sprites_mirrored)]
        obstacles.append(
            Obstacle(
                sprite=sprite,
                xy=(SCENE_WIDTH - sprite.get_width(), cursor)
            )
        )
        cursor += sprite.get_height()
        i += 1
    # endregion

    # region Generate obstacles
    max_obstacle_height = max(
        sprite.get_height()
        for sprite in media.obstacle_sprites
    )

    row_height = round(max_obstacle_height * 1.3)
    padding = round((row_height - max_obstacle_height) / 2)

    skier_width = calculate_skier_width(media)

    margin_width = max(
        sprite.get_width()
        for sprite in media.margin_sprites
    )

    # Skp the first row so that the player can get prepared
    cursor = row_height
    while cursor < SCENE_HEIGHT:
        obstacle_count = random.randint(1, 4)
        last_x = None  # type: Optional[int]
        for _ in range(obstacle_count):
            if last_x is None:
                x = margin_width + random.randint(0, 2 * skier_width)
            else:
                x = last_x + random.randint(2 * skier_width, 4 * skier_width)

            # Add some jitter to y to make the level more natural
            y = cursor + random.randint(0, padding)

            obstacle_sprite = random.choice(media.obstacle_sprites)

            # We can not display this obstacle, so we are done for
            # the row.
            if x + obstacle_sprite.get_width() >= SCENE_WIDTH - margin_width:
                break

            obstacles.append(
                Obstacle(
                    sprite=random.choice(media.obstacle_sprites),
                    xy=(x, y)
                )
            )
    # endregion

    return Level(obstacles=obstacles)


class SkierAction(enum.Enum):
    """Represent the action of the skier."""
    FORWARD = 0
    LEFT = 1
    RIGHT = 2


class Skier:
    """Capture the state of the skier."""

    #: In world coordinates, center of the skier.
    xy: Tuple[int, int]

    action: SkierAction

    def __init__(self, xy: Tuple[int, int], action: SkierAction) -> None:
        """Initialize with the given values."""
        self.xy = xy
        self.action = action


def skier_action_to_sprite(
        action: SkierAction,
        media: Media
) -> pygame.surface.Surface:
    """Determine the sprite corresponding to the skier action."""
    if action is SkierAction.LEFT:
        return media.skier_left_sprite
    elif action is SkierAction.RIGHT:
        return media.skier_left_sprite
    elif action is SkierAction.FORWARD:
        return media.skier_forward_sprite
    else:
        common.assert_never(action)

def skier_bounding_box(
        xy: Tuple[int, int],
        skier_sprite: pygame.surface.Surface
) -> Tuple[int, int, int, int]:
    """
    Compute (xmin, ymin, xmax, ymax) based on the center (x, y).
    
    In world coordinates.
    """
    sprite_height = skier_sprite.get_height()
    sprite_width = skier_sprite.get_width()
    
    if sprite_height % 2 == 0:
        ymin = xy[1] - sprite_height // 2 - 1
        ymax = xy[1] + sprite_height // 2
    else:
        ymin = xy[1] - sprite_height // 2
        ymax = xy[1] + sprite_height // 2
        
    assert ymax - ymin == sprite_height

    if sprite_width % 2 == 0:
        xmin = xy[0] - sprite_width // 2 - 1
        xmax = xy[0] + sprite_width // 2
    else:
        xmin = xy[0] - sprite_width // 2
        xmax = xy[0] + sprite_width // 2

    assert xmax - xmin == sprite_width

    return xmin, ymin, xmax, ymax
    


class GameOverKind(enum.Enum):
    """Capture how the game ended."""
    SUCCESS = 0
    CRASH = 1


class GameOver:
    """Capture how and when the game ended."""
    timestamp: float

    kind: GameOverKind

    def __init__(self, timestamp: float, kind: GameOverKind) -> None:
        """Initialize with the given values."""
        self.timestamp = timestamp
        self.kind = kind


class State:
    """Represent the state of the game."""
    #: Set if we received the signal to quit the game
    received_quit: bool

    #: Timestamp when the game started, seconds since epoch
    game_start: float

    #: Current clock in the game, seconds since epoch
    now: float

    #: Set when the game finishes
    game_over: Optional[GameOver]

    level_id: int
    level: Level

    skier: Skier

    def __init__(self, game_start: float, media: Media) -> None:
        """Initialize with the given values and the defaults."""
        initialize_state(self, game_start, media)


def initialize_state(state: State, game_start: float, media: Media) -> None:
    """Initialize the state to the start one."""
    state.received_quit = False
    state.game_start = game_start
    state.now = game_start
    state.game_over = None

    state.level_id = 0
    state.level = generate_level(media=media)

    skier_height = calculate_skier_height(media)
    state.skier = Skier(
        xy=(round(SCENE_WIDTH / 2), round(skier_height / 2)),
        action=SkierAction.FORWARD
    )


LEVEL_COUNT = 5


@require(lambda xmin_a, xmax_a: xmin_a <= xmax_a)
@require(lambda ymin_a, ymax_a: ymin_a <= ymax_a)
@require(lambda xmin_b, xmax_b: xmin_b <= xmax_b)
@require(lambda ymin_b, ymax_b: ymin_b <= ymax_b)
def intersect(
        xmin_a: Union[int, float],
        ymin_a: Union[int, float],
        xmax_a: Union[int, float],
        ymax_a: Union[int, float],
        xmin_b: Union[int, float],
        ymin_b: Union[int, float],
        xmax_b: Union[int, float],
        ymax_b: Union[int, float],
) -> bool:
    """Return true if the two bounding boxes intersect."""
    return (xmin_a <= xmax_b and xmax_a >= xmin_b) and (
            ymin_a <= ymax_b and ymax_a >= ymin_b
    )

def update_state_on_tick(state: State, now: float, media: Media) -> None:
    """Update state on one game cycle."""
    time_delta = now - state.now
    
    # region Check if the skier hit an obstacle
    skier_sprite = skier_action_to_sprite(state.skier.action, media)
    
    
    
    skier_xmin =  
    
    
    # endregion
    
    

def main() -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()

    return 0


if __name__ == "__main__":
    sys.exit(main())
