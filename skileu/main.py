"""Ski through a dangerous forest."""
import abc
import argparse
import enum
import fractions
import importlib
import itertools
import os
import pathlib
import random
import sys
import time
from typing import List, Tuple, Optional, Union, Final, Protocol

import cv2
import pygame
import pygame.freetype
from icontract import require, ensure

import skileu
from skileu import common, bodypose

assert skileu.__doc__ == __doc__

PACKAGE_DIR = (
    pathlib.Path(str(importlib.resources.files(__package__)))  # type: ignore
    if __package__ is not None
    else pathlib.Path(os.path.realpath(__file__)).parent
)


class MaskedSprite:
    """Represent a sprite with a bitmask."""

    sprite: Final[pygame.surface.Surface]
    mask: Final[pygame.mask.Mask]

    def __init__(self, sprite: pygame.surface.Surface) -> None:
        """Initialize with the given values."""
        self.sprite = sprite
        self.mask = pygame.mask.from_surface(sprite)

    def get_height(self) -> int:
        """Return the height of the sprite."""
        return self.sprite.get_height()

    def get_width(self) -> int:
        """Return the width of the sprite."""
        return self.sprite.get_width()

    def get_size(self) -> Tuple[int, int]:
        """Return the size of the sprite."""
        return self.get_size()


def make_non_transparent_light_gray(sprite: pygame.surface.Surface) -> None:
    """Set all the non-transparent pixels to light gray."""
    for y in range(sprite.get_height()):
        for x in range(sprite.get_width()):
            xy = (x, y)
            pixel = sprite.get_at(xy)
            if pixel[3] != 0:
                sprite.set_at(xy, (200, 200, 200, 255))


class SkierSpriteSet:
    """Capture the appearance of the skier."""

    left: Final[MaskedSprite]
    right: Final[MaskedSprite]
    forward: Final[MaskedSprite]
    speed_up: Final[MaskedSprite]

    left_trail: Final[pygame.surface.Surface]
    right_trail: Final[pygame.surface.Surface]
    forward_trail: Final[pygame.surface.Surface]
    speed_up_trail: Final[pygame.surface.Surface]

    max_height: Final[int]
    max_width: Final[int]

    def __init__(
        self,
        left: MaskedSprite,
        right: MaskedSprite,
        forward: MaskedSprite,
        speed_up: MaskedSprite,
    ) -> None:
        """Initialize with the given values."""
        self.left = left
        self.right = right
        self.forward = forward
        self.speed_up = speed_up

        max_width = 0
        max_height = 0
        for masked_sprite in [left, right, forward, speed_up]:
            max_width = max(max_width, masked_sprite.get_width())
            max_height = max(max_height, masked_sprite.get_height())

        self.max_width = max_width
        self.max_height = max_height

        self.left_trail = left.sprite.subsurface(
            (0, left.get_height() - 4, left.get_width(), 4)
        )
        make_non_transparent_light_gray(self.left_trail)

        self.right_trail = right.sprite.subsurface(
            (0, right.get_height() - 4, right.get_width(), 4)
        )
        make_non_transparent_light_gray(self.right_trail)

        self.forward_trail = forward.sprite.subsurface(
            (0, forward.get_height() - 4, forward.get_width(), 4)
        )
        make_non_transparent_light_gray(self.forward_trail)

        self.speed_up_trail = speed_up.sprite.subsurface(
            (0, speed_up.get_height() - 4, speed_up.get_width(), 4)
        )
        make_non_transparent_light_gray(self.speed_up_trail)


class ActorSpriteSet:
    """Represent actor sprites."""

    #: Sprites of the actor idling around, to the left
    idle_left: Final[List[MaskedSprite]]

    #: Sprites of the actor idling around, to the right
    idle_right: Final[List[MaskedSprite]]

    #: Sprites of the actor walking to the side, to the left
    walk_left: Final[List[MaskedSprite]]

    #: Sprites of the actor walking to the side, to the right
    walk_right: Final[List[MaskedSprite]]

    #: Maximum width over all the action sprite sets
    max_width: Final[int]

    #: Maximum height over all the action sprite sets
    max_height: Final[int]

    def __init__(
        self, idle_left: List[MaskedSprite], walk_left: List[MaskedSprite]
    ) -> None:
        """Initialize with the given values."""
        self.idle_left = idle_left
        self.idle_right = [
            MaskedSprite(pygame.transform.flip(masked_sprite.sprite, True, False))
            for masked_sprite in idle_left
        ]

        self.walk_left = walk_left
        self.walk_right = [
            MaskedSprite(pygame.transform.flip(masked_sprite.sprite, True, False))
            for masked_sprite in walk_left
        ]

        self.max_width = max(
            sprite.get_width() for sprite in self.idle_left + self.walk_left
        )

        self.max_height = max(
            sprite.get_height() for sprite in self.idle_left + self.walk_left
        )


class Media:
    """Represent all the media loaded in the main memory from the file system."""

    def __init__(
        self,
        skier_sprite_set: SkierSpriteSet,
        obstacle_sprites: List[MaskedSprite],
        margin_sprites: List[MaskedSprite],
        actor_sprite_sets: List[ActorSpriteSet],
        font: pygame.freetype.Font,  # type: ignore
        success_music_path: pathlib.Path,
    ) -> None:
        """Initialize with the given values."""
        self.skier_sprite_set = skier_sprite_set
        self.obstacle_sprites = obstacle_sprites
        self.margin_sprites = margin_sprites
        self.margin_sprites_mirrored = [
            MaskedSprite(pygame.transform.flip(masked_sprite.sprite, True, False))
            for masked_sprite in margin_sprites
        ]
        self.actor_sprite_sets = actor_sprite_sets
        self.font = font
        self.success_music_path = success_music_path


SCENE_WIDTH = 800
SCENE_HEIGHT = 600

ROAD_MARGIN = 128


@require(lambda sprite_width: sprite_width > 0)
@require(lambda sprite_height: sprite_height > 0)
@require(lambda sprite_width, sprite_set: sprite_set.get_width() % sprite_width == 0)
@require(lambda sprite_height, sprite_set: sprite_set.get_height() % sprite_height == 0)
@ensure(
    lambda sprite_width, sprite_height, result: all(
        sprite.get_width() == sprite_width and sprite.get_height() == sprite_height
        for row in result
        for sprite in row
    )
)
def cut_out_sprite_set(
    sprite_set: pygame.surface.Surface, sprite_width: int, sprite_height: int
) -> List[List[pygame.surface.Surface]]:
    """Cut out a sprite set kept in a single image."""
    rows = sprite_set.get_height() // sprite_height
    columns = sprite_set.get_width() // sprite_width

    table = []  # type: List[List[pygame.surface.Surface]]

    for row_i in range(rows):
        row = []  # type: List[pygame.surface.Surface]
        for column_i in range(columns):
            x = column_i * sprite_width
            y = row_i * sprite_height

            sprite = sprite_set.subsurface((x, y, sprite_width, sprite_height))
            row.append(sprite)

        table.append(row)

    return table


@ensure(lambda result: (result[0] is not None) ^ (result[1] is not None))
def load_media() -> Tuple[Optional[Media], Optional[str]]:
    """Load the media from the file system."""
    images_dir = PACKAGE_DIR / "media/images"

    pth = images_dir / "skier-forward.png"
    try:
        skier_forward_sprite = pygame.image.load(str(pth)).convert_alpha()
    except Exception as exception:
        return None, f"Failed to load {pth}: {exception}"

    pth = images_dir / "skier-speed-up.png"
    try:
        skier_speed_up_sprite = pygame.image.load(str(pth)).convert_alpha()
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

    actor_dir = PACKAGE_DIR / "media/images/actors"
    actor_paths_and_dimensions = [
        (actor_dir / "boar", (64, 40)),
        (actor_dir / "deer", (72, 52)),
        (actor_dir / "fox", (64, 36)),
        (actor_dir / "rabbit", (32, 26)),
        (actor_dir / "wolf", (64, 40)),
    ]

    actor_sprite_sets = []  # type: List[ActorSpriteSet]
    for pth, (sprite_width, sprite_height) in actor_paths_and_dimensions:
        idle_pth = pth / "idle.png"
        if not idle_pth.exists():
            return None, f"The sprite set does not exist: {idle_pth}"

        try:
            sprite_set = pygame.image.load(str(idle_pth)).convert_alpha()
            sprite_table = cut_out_sprite_set(
                sprite_set=sprite_set,
                sprite_width=sprite_width,
                sprite_height=sprite_height,
            )
            idle_sprites = sprite_table[0]
        except Exception as exception:
            return None, f"Failed to load the sprite set {idle_pth}: {exception}"

        walk_pth = pth / "walk.png"
        if not walk_pth.exists():
            return None, f"The sprite set does not exist: {walk_pth}"

        try:
            sprite_set = pygame.image.load(str(walk_pth)).convert_alpha()
            sprite_table = cut_out_sprite_set(
                sprite_set=sprite_set,
                sprite_width=sprite_width,
                sprite_height=sprite_height,
            )
            walk_sprites = sprite_table[0]
        except Exception as exception:
            return None, f"Failed to load the sprite set {walk_pth}: {exception}"

        actor_sprite_sets.append(
            ActorSpriteSet(
                idle_left=[MaskedSprite(sprite) for sprite in idle_sprites],
                walk_left=[MaskedSprite(sprite) for sprite in walk_sprites],
            )
        )

    pth = PACKAGE_DIR / "media/fonts/freesansbold.ttf"
    try:
        font = pygame.freetype.Font(str(pth))  # type: ignore
    except Exception as exception:
        return None, f"Failed to load {pth}: {exception}"

    success_music_path = PACKAGE_DIR / "media/music/success.mid"
    if not pth.exists():
        return None, f"File does not exist: {success_music_path}"

    return (
        Media(
            SkierSpriteSet(
                left=MaskedSprite(skier_left_sprite),
                right=MaskedSprite(skier_right_sprite),
                forward=MaskedSprite(skier_forward_sprite),
                speed_up=MaskedSprite(skier_speed_up_sprite),
            ),
            obstacle_sprites=[MaskedSprite(sprite) for sprite in obstacle_sprites]
            + [
                MaskedSprite(pygame.transform.flip(sprite, True, False))
                for sprite in obstacle_sprites
            ],
            margin_sprites=[MaskedSprite(sprite) for sprite in margin_sprites],
            actor_sprite_sets=actor_sprite_sets,
            font=font,
            success_music_path=success_music_path,
        ),
        None,
    )


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


def calculate_bounding_box(
    xy: Tuple[float, float], masked_sprite: MaskedSprite
) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box of an object.

    ``xy`` denotes the top-left corner, in screen coordinates.
    """
    return (
        xy[0],
        xy[1],
        xy[0] + masked_sprite.get_width(),
        xy[1] + masked_sprite.get_height(),
    )


def calculate_collision(
    bounding_box: Tuple[float, float, float, float],
    mask: pygame.mask.Mask,
    other_bounding_box: Tuple[float, float, float, float],
    other_mask: pygame.mask.Mask,
) -> Optional[Tuple[float, float]]:
    """Calculate the position of the collision with another object, if any."""
    if intersect(*bounding_box, *other_bounding_box):
        xmin = bounding_box[0]
        ymin = bounding_box[1]

        other_xmin = other_bounding_box[0]
        other_ymin = other_bounding_box[1]

        offset = (round(other_xmin - xmin), round(other_ymin - ymin))

        collision_xy = mask.overlap(other_mask, offset)

        if collision_xy is not None:
            return collision_xy[0] + xmin, collision_xy[1] + ymin

    return None


class ColliderProtocol(Protocol):
    """Represent a class of collider objects or actors."""

    #: Top-left corner in screen coordinates
    xy: Tuple[float, float]

    @abc.abstractmethod
    def determine_masked_sprite(self, now: float) -> MaskedSprite:
        """Determine the sprite given the timestamp."""
        raise NotImplementedError()


class Obstacle(ColliderProtocol):
    """Represent a static obstacle in the level."""

    #: Appearance
    masked_sprite: Final[MaskedSprite]

    #: Top-left corner in screen coordinates
    xy: Tuple[float, float]

    def __init__(self, masked_sprite: MaskedSprite, xy: Tuple[float, float]) -> None:
        """Initialize with the given values."""
        self.masked_sprite = masked_sprite
        self.xy = xy

    def determine_masked_sprite(self, now: float) -> MaskedSprite:
        return self.masked_sprite


class Actor(ColliderProtocol):
    """Represent a non-player character in the level."""

    #: Appearance
    sprite_set: Final[ActorSpriteSet]

    #: Top-left corner in screen coordinates
    xy: Tuple[float, float]

    #: Always positive, or 0 if idling
    velocity: float

    #: <=0 => left, >1 => right
    direction: int

    action_start: float
    action_eta: float

    def __init__(
        self,
        sprite_set: ActorSpriteSet,
        xy: Tuple[float, float],
        velocity: float,
        direction: int,
        action_start: float,
        action_eta: float,
    ) -> None:
        """Initialize with the given values."""
        self.sprite_set = sprite_set
        self.xy = xy
        self.velocity = velocity
        self.direction = direction
        self.action_start = action_start
        self.action_eta = action_eta

    def determine_masked_sprite(self, now: float) -> MaskedSprite:
        """Determine the sprite given the timestamp."""
        # In seconds
        animation_lambda = 0.15

        i = round((now - self.action_start) / animation_lambda)

        if self.velocity == 0:
            if self.direction <= 0:
                return self.sprite_set.idle_left[i % len(self.sprite_set.idle_left)]
            else:
                return self.sprite_set.idle_right[i % len(self.sprite_set.idle_right)]
        else:
            if self.direction <= 0:
                return self.sprite_set.walk_left[i % len(self.sprite_set.walk_left)]
            else:
                return self.sprite_set.walk_right[i % len(self.sprite_set.walk_right)]


class Level:
    """Represent a single scene."""

    obstacles: List[Obstacle]
    actors: List[Actor]

    snow: pygame.surface.Surface

    def __init__(self, obstacles: List[Obstacle], actors: List[Actor]) -> None:
        """Initialize with the given values."""
        self.obstacles = obstacles
        self.actors = actors

        self.snow = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
        self.snow.fill((255, 255, 255))


def generate_level(now: float, media: Media) -> Level:
    """Generate randomly a level."""
    obstacles = []  # type: List[Obstacle]
    actors = []  # type: List[Actor]

    assert all(sprite.get_height() > 0 for sprite in media.margin_sprites), (
        "All margin sprites at least 1 pixel tall "
        "so that we do not enter an endless loop"
    )

    # region Generate left margin
    cursor = 0.0
    while cursor < SCENE_HEIGHT:
        masked_sprite = random.choice(media.margin_sprites)
        obstacles.append(Obstacle(masked_sprite=masked_sprite, xy=(0, cursor)))
        cursor += masked_sprite.get_height()
    # endregion

    # region Generate right margin
    cursor = 0
    while cursor < SCENE_HEIGHT:
        masked_sprite = random.choice(media.margin_sprites_mirrored)
        obstacles.append(
            Obstacle(
                masked_sprite=masked_sprite,
                xy=(SCENE_WIDTH - masked_sprite.get_width(), cursor),
            )
        )
        cursor += masked_sprite.get_height()
    # endregion

    # region Generate obstacles and actors
    assert all(sprite.get_height() > 0 for sprite in media.obstacle_sprites), (
        "All obstacle sprites at least 1 pixel tall "
        "so that we do not enter an endless loop"
    )

    max_obstacle_and_actor_height = max(
        max(sprite.get_height() for sprite in media.obstacle_sprites),
        max(
            actor_sprite_set.max_height for actor_sprite_set in media.actor_sprite_sets
        ),
    )

    row_height = round(max_obstacle_and_actor_height * 2.3)
    padding = round((row_height - max_obstacle_and_actor_height) / 2)

    skier_width = media.skier_sprite_set.max_width

    margin_width = max(sprite.get_width() for sprite in media.margin_sprites)

    # Skp the first row so that the player can get prepared
    clearance_line_at_start = (
        SCENE_HEIGHT
        - 3 * media.skier_sprite_set.max_height
        - max_obstacle_and_actor_height
    )

    cursor = 0
    while cursor < clearance_line_at_start:
        coin = random.random()

        if coin < 0.7:
            next_x_start = margin_width
            while True:
                obstacle_sprite = random.choice(media.obstacle_sprites)

                x = next_x_start + random.randint(
                    round(1.1 * skier_width), 2 * skier_width
                )

                # Add some jitter to y to make the level more natural
                y = min(
                    clearance_line_at_start,
                    cursor + random.randint(0, padding),
                )

                # We can not display this obstacle, so we are done for
                # the row.
                if x + obstacle_sprite.get_width() >= SCENE_WIDTH - margin_width:
                    break

                obstacles.append(Obstacle(masked_sprite=obstacle_sprite, xy=(x, y)))
                next_x_start = x + obstacle_sprite.get_width()

            cursor += row_height
        else:
            actor_sprite_set = random.choice(media.actor_sprite_sets)

            actor_first_x = margin_width
            actor_last_x = SCENE_WIDTH - margin_width - actor_sprite_set.max_width

            y = min(
                clearance_line_at_start,
                cursor + random.randint(0, padding),
            )

            actor = Actor(
                sprite_set=actor_sprite_set,
                xy=(random.randint(actor_first_x, actor_last_x), y),
                velocity=0,
                direction=random.randint(0, 1),
                action_start=now,
                # We will immediately pick a random action at the next tick.
                # This makes the logic of level generation much simpler.
                action_eta=now,
            )

            actors.append(actor)

            cursor += max_obstacle_and_actor_height * 1.5
    # endregion

    return Level(obstacles=obstacles, actors=actors)


class SkierAction(enum.Enum):
    """Represent the action of the skier."""

    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    SPEED_UP = 3


class Skier:
    """Capture the state of the skier."""

    #: Center of the skier in screen coordinates
    center_xy: Tuple[float, float]

    action: SkierAction

    #: Possible appearance
    skier_sprite_set: Final[SkierSpriteSet]

    velocity_factor: float

    def __init__(
        self,
        center_xy: Tuple[float, float],
        action: SkierAction,
        skier_sprite_set: SkierSpriteSet,
        velocity_factor: float,
    ) -> None:
        """Initialize with the given values."""
        self.center_xy = center_xy
        self.action = action
        self.skier_sprite_set = skier_sprite_set
        self.velocity_factor = velocity_factor

    def determine_masked_sprite(self) -> MaskedSprite:
        """Determine the sprite of the skier."""
        if self.action is SkierAction.LEFT:
            return self.skier_sprite_set.left
        elif self.action is SkierAction.RIGHT:
            return self.skier_sprite_set.right
        elif self.action is SkierAction.FORWARD:
            return self.skier_sprite_set.forward
        elif self.action is SkierAction.SPEED_UP:
            return self.skier_sprite_set.speed_up
        else:
            common.assert_never(self.action)

    def determine_trail_sprite(self) -> pygame.surface.Surface:
        """Determine the sprite for the trail."""
        if self.action is SkierAction.LEFT:
            return self.skier_sprite_set.left_trail
        elif self.action is SkierAction.RIGHT:
            return self.skier_sprite_set.right_trail
        elif self.action is SkierAction.FORWARD:
            return self.skier_sprite_set.forward_trail
        elif self.action is SkierAction.SPEED_UP:
            return self.skier_sprite_set.speed_up_trail
        else:
            common.assert_never(self.action)


def calculate_xy_from_center(
    center_xy: Tuple[float, float], masked_sprite: MaskedSprite
) -> Tuple[float, float]:
    """Calculate the top-left corner in screen coordinates."""
    return (
        center_xy[0] - masked_sprite.get_width() / 2,
        center_xy[1] - masked_sprite.get_height() / 2,
    )


class GameOver:
    """Capture how and when the game ended."""

    timestamp: float

    def __init__(self, timestamp: float) -> None:
        """Initialize with the given values."""
        self.timestamp = timestamp


class GameOverOk(GameOver):
    """Represent a successful game."""


class Collision:
    """Capture a collision between a skier and something."""

    def __init__(
        self,
        skier_sprite: pygame.surface.Surface,
        skier_xy: Tuple[float, float],
        collider_sprite: pygame.surface.Surface,
        collider_xy: Tuple[float, float],
        xy: Tuple[float, float],
    ) -> None:
        """
        Initialize with the given values.

        All coordinates in world coordinates.
        """
        self.skier_sprite = skier_sprite
        self.skier_xy = skier_xy
        self.collider_sprite = collider_sprite
        self.collider_xy = collider_xy
        self.xy = xy


class GameOverCrash(GameOver):
    """Capture the game over upon collision between a skier and something."""

    #: Obstacle or actor that the skier had a collision with
    collider: ColliderProtocol

    collision: Collision

    def __init__(self, timestamp: float, collision: Collision) -> None:
        """Initialize with the given values."""
        GameOver.__init__(self, timestamp)
        self.collision = collision


class State:
    """Represent the state of the game."""

    #: Set if we received the signal to quit the game
    received_quit: bool

    #: Timestamp when the game started, seconds since epoch
    game_start: float

    #: Current clock in the game, seconds since epoch
    now: float

    #: Set when the game finishes
    game_over: Optional[Union[GameOverOk, GameOverCrash]]

    level_id: int
    level: Level

    skier: Skier

    def __init__(self, game_start: float, media: Media, level: Level) -> None:
        """Initialize with the given values and the defaults."""
        initialize_state(self, game_start, media, level=level)


def initialize_state(
    state: State, game_start: float, media: Media, level: Level
) -> None:
    """Initialize the state to the start one."""
    state.received_quit = False
    state.game_start = game_start
    state.now = game_start
    state.game_over = None

    state.level_id = 0
    state.level = level

    skier_height = media.skier_sprite_set.max_height
    state.skier = Skier(
        center_xy=(round(SCENE_WIDTH / 2), SCENE_HEIGHT - skier_height / 2),
        action=SkierAction.FORWARD,
        skier_sprite_set=media.skier_sprite_set,
        velocity_factor=1.0,
    )


LEVEL_COUNT = 5

#: Allow for global ad-hoc tweaks in case some players are too slow
GLOBAL_VELOCITY_FACTOR = 0.75

#: Velocity in screen coordinates depending on action, (x, y)
VELOCITY_DISPATCH = {
    SkierAction.FORWARD: (0, -GLOBAL_VELOCITY_FACTOR * 50),
    # NOTE (mristin, 2023-03-18):
    # Speed up is simply for the appearance — the local velocity factor is determined
    # continuously, not discretely.
    SkierAction.SPEED_UP: (0, -GLOBAL_VELOCITY_FACTOR * 50),
    SkierAction.LEFT: (-50, -GLOBAL_VELOCITY_FACTOR * 30),
    SkierAction.RIGHT: (50, -GLOBAL_VELOCITY_FACTOR * 30),
}
assert all(action in VELOCITY_DISPATCH for action in SkierAction)

#: In px/second
ACTOR_VELOCITY = 50


def update_state_on_tick(state: State, now: float, media: Media) -> None:
    """Update state on one game cycle."""
    time_delta = now - state.now

    state.now = now

    if state.game_over is not None:
        return

    # region Check for collision(s) between obstacles, actors and the skier
    skier_masked_sprite = state.skier.determine_masked_sprite()
    skier_xy = calculate_xy_from_center(state.skier.center_xy, skier_masked_sprite)
    skier_bbox = calculate_bounding_box(skier_xy, skier_masked_sprite)

    for collider in itertools.chain(state.level.obstacles, state.level.actors):
        collider_masked_sprite = collider.determine_masked_sprite(state.now)
        collider_bbox = calculate_bounding_box(collider.xy, collider_masked_sprite)

        collision_xy = calculate_collision(
            skier_bbox,
            skier_masked_sprite.mask,
            collider_bbox,
            collider_masked_sprite.mask,
        )

        if collision_xy is not None:
            state.game_over = GameOverCrash(
                timestamp=now,
                collision=Collision(
                    skier_sprite=skier_masked_sprite.sprite,
                    skier_xy=skier_xy,
                    collider_sprite=collider_masked_sprite.sprite,
                    collider_xy=collider.xy,
                    xy=collision_xy,
                ),
            )
            return
    # endregion

    # region Check for reaching the end of level
    if skier_xy[1] <= 0:
        if state.level_id == LEVEL_COUNT - 1:
            state.game_over = GameOverOk(now)
        else:
            state.level_id += 1
            state.level = generate_level(now=state.now, media=media)

            skier_height = state.skier.skier_sprite_set.max_height
            state.skier.center_xy = (
                state.skier.center_xy[0],
                SCENE_HEIGHT - skier_height / 2,
            )

        return
    # endregion

    # region Leave the trail
    trail_sprite = state.skier.determine_trail_sprite()
    trail_xy = (skier_xy[0], skier_xy[1] + skier_masked_sprite.get_height() - 1)
    state.level.snow.blit(trail_sprite, trail_xy)
    # endregion

    # region Update skier
    velocity = VELOCITY_DISPATCH[state.skier.action]

    state.skier.center_xy = (
        state.skier.center_xy[0]
        + state.skier.velocity_factor * velocity[0] * time_delta,
        state.skier.center_xy[1]
        + state.skier.velocity_factor * velocity[1] * time_delta,
    )
    # endregion

    # region Update actors
    for actor in state.level.actors:
        if state.now > actor.action_eta:
            actor.action_start = state.now
            actor.action_eta = actor.action_start + random.random() * 4

            action_coin = random.random()
            direction = random.choice([-1, 1])
            if action_coin < 0.1:
                actor.velocity = 0
            else:
                actor.velocity = ACTOR_VELOCITY

            actor.direction = direction
        else:
            new_x = actor.xy[0] + actor.direction * actor.velocity * time_delta
            if (
                new_x < ROAD_MARGIN
                or new_x + actor.sprite_set.max_width >= SCENE_WIDTH - ROAD_MARGIN
            ):
                # NOTE (mristin, 2023-03-07):
                # The actor reached the end of the trail, and needs to change
                # the direction.
                actor.direction *= -1
            else:
                actor.xy = (new_x, actor.xy[1])

    # endregion


def cvmat_to_surface(image: cv2.Mat) -> pygame.surface.Surface:
    """Convert from OpenCV to pygame."""
    height, width, _ = image.shape
    if height == 0 and width == 0:
        return pygame.surface.Surface((1, 1))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image_rgb.tobytes(), (width, height), "RGB")


def recognize_action_from_detection(
    detection: bodypose.Detection, frame: cv2.Mat
) -> Tuple[Optional[SkierAction], float, pygame.surface.Surface]:
    """
    Infer the action based on the body pose detection.

    Return (action, velocity factor) and the camera frame with the body wire
    illustrating it.
    """
    frame_height, frame_width, _ = frame.shape

    left_hip = detection.keypoints.get(skileu.bodypose.KeypointLabel.LEFT_HIP, None)

    right_hip = detection.keypoints.get(skileu.bodypose.KeypointLabel.RIGHT_HIP, None)

    left_knee = detection.keypoints.get(skileu.bodypose.KeypointLabel.LEFT_KNEE, None)

    right_knee = detection.keypoints.get(skileu.bodypose.KeypointLabel.RIGHT_KNEE, None)

    left_ankle = detection.keypoints.get(skileu.bodypose.KeypointLabel.LEFT_ANKLE, None)

    right_ankle = detection.keypoints.get(
        skileu.bodypose.KeypointLabel.RIGHT_ANKLE, None
    )

    left_wrist = detection.keypoints.get(skileu.bodypose.KeypointLabel.LEFT_WRIST, None)

    right_wrist = detection.keypoints.get(
        skileu.bodypose.KeypointLabel.RIGHT_WRIST, None
    )

    frame_with_wire = frame.copy()

    action = None  # type: Optional[SkierAction]
    if (
        left_hip is not None
        and right_hip is not None
        and left_knee is not None
        and right_knee is not None
        and left_ankle is not None
        and right_ankle is not None
    ):
        hip = (
            round(frame_width * (left_hip.x + right_hip.x) / 2.0),
            round(frame_height * (left_hip.y + right_hip.y) / 2.0),
        )

        knee = (
            round(frame_width * (left_knee.x + right_knee.x) / 2.0),
            round(frame_height * (left_knee.y + right_knee.y) / 2.0),
        )

        ankle = (
            round(frame_width * (left_ankle.x + right_ankle.x) / 2.0),
            round(frame_height * (left_ankle.y + right_ankle.y) / 2.0),
        )

        # region Draw the relevant keypoints
        cv2.line(frame_with_wire, hip, knee, (255, 255, 255), 10)
        cv2.line(frame_with_wire, knee, ankle, (255, 255, 255), 10)

        cv2.circle(frame_with_wire, hip, 20, (0, 0, 255), -1)
        cv2.circle(frame_with_wire, knee, 20, (0, 0, 255), -1)
        cv2.circle(frame_with_wire, ankle, 20, (0, 0, 255), -1)
        # endregion

        action = SkierAction.FORWARD
        if hip[1] < ankle[1] and knee[1] < ankle[1]:
            angle = bodypose.compute_knee_angle(
                (hip[0], frame_height - hip[1]),
                (knee[0], frame_height - knee[1]),
                (ankle[0], frame_height - ankle[1]),
            )

            if abs(angle) < 150:
                action = SkierAction.RIGHT if angle < 0 else SkierAction.LEFT

    velocity_factor = 1.0
    if (
        left_wrist is not None
        and right_wrist is not None
        and left_knee is not None
        and right_knee is not None
        and left_ankle is not None
        and right_ankle is not None
    ):
        knee = (
            round(frame_width * (left_knee.x + right_knee.x) / 2.0),
            round(frame_height * (left_knee.y + right_knee.y) / 2.0),
        )

        ankle = (
            round(frame_width * (left_ankle.x + right_ankle.x) / 2.0),
            round(frame_height * (left_ankle.y + right_ankle.y) / 2.0),
        )

        wrist = (
            round(frame_width * (left_wrist.x + right_wrist.x) / 2.0),
            round(frame_height * (left_wrist.y + right_wrist.y) / 2.0),
        )

        cv2.circle(frame_with_wire, wrist, 20, (255, 255, 0), -1)

        total = ankle[1] - knee[1]
        if total == 0:
            ratio = 0.0
        else:
            ratio = min(1.0, max(0.0, (ankle[1] - wrist[1]) / total))

        # NOTE (mristin, 2023-03-08):
        # This is an arbitrary equation that seemed to work well in the game play.
        velocity_factor = (2.0 - ratio) ** 1.7

        if ratio < 0.7 and action is SkierAction.FORWARD:
            action = SkierAction.SPEED_UP

    return action, velocity_factor, cvmat_to_surface(frame_with_wire)


def draw_obstacle_on_scene(scene: pygame.surface.Surface, obstacle: Obstacle) -> None:
    """Draw the obstacle on the scene."""
    scene.blit(obstacle.masked_sprite.sprite, obstacle.xy)


def draw_skier_on_scene(scene: pygame.surface.Surface, skier: Skier) -> None:
    """Draw the skier on the scene."""
    masked_sprite = skier.determine_masked_sprite()
    xy = calculate_xy_from_center(skier.center_xy, masked_sprite)

    scene.blit(masked_sprite.sprite, xy)


def draw_actor_on_scene(
    scene: pygame.surface.Surface, actor: Actor, now: float
) -> None:
    """Draw the actor on the scene."""
    masked_sprite = actor.determine_masked_sprite(now)

    scene.blit(masked_sprite.sprite, actor.xy)


@require(lambda state: state.game_over is None)
def render_in_game(
    state: State, media: Media, frame_with_wire: pygame.surface.Surface
) -> pygame.surface.Surface:
    """Render the game screen based on the state."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.blit(state.level.snow, (0, 0))

    for obstacle in state.level.obstacles:
        draw_obstacle_on_scene(scene, obstacle)

    for actor in state.level.actors:
        draw_actor_on_scene(scene, actor, state.now)

    draw_skier_on_scene(scene, state.skier)

    fog = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT)).convert_alpha()
    fog.fill((255, 255, 255, 220))

    pygame.draw.circle(fog, (255, 255, 255, 150), state.skier.center_xy, 160, width=10)
    pygame.draw.circle(fog, (255, 255, 255, 100), state.skier.center_xy, 150, width=10)
    pygame.draw.circle(fog, (255, 255, 255, 50), state.skier.center_xy, 140, width=10)
    pygame.draw.circle(fog, (255, 255, 255, 25), state.skier.center_xy, 130, width=10)
    pygame.draw.circle(fog, (0, 0, 0, 0), state.skier.center_xy, 130)
    scene.blit(fog, (0, 0))

    frame_with_wire_resized = pygame.surface.Surface((120, 120))
    resize_image_to_canvas_and_blit(frame_with_wire, frame_with_wire_resized)
    scene.blit(
        frame_with_wire_resized,
        (0, SCENE_HEIGHT - frame_with_wire_resized.get_height()),
    )

    media.font.render_to(
        scene,
        (ROAD_MARGIN + 10, 10),
        'Press "q" to quit and "r" to restart',
        (0, 0, 0),
        size=12,
    )

    media.font.render_to(
        scene,
        (10, 10),
        f"Level: {state.level_id + 1} / {LEVEL_COUNT}",
        (255, 255, 255),
        size=16,
    )

    return scene


def render_game_over(state: State, media: Media) -> pygame.surface.Surface:
    """Render the "game over" dialogue as a scene."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((255, 255, 255))

    assert state.game_over is not None

    if isinstance(state.game_over, GameOverOk):
        road_length = (state.level_id + 1) * SCENE_HEIGHT
        time_delta = state.game_over.timestamp - state.game_start
        average_velocity = road_length / time_delta
        media.font.render_to(scene, (20, 20), "You made it!", (0, 0, 0), size=16)

        media.font.render_to(
            scene,
            (20, 60),
            f"Average velocity: {average_velocity:.1f} pixels / second",
            (0, 0, 0),
            size=16,
        )

        media.font.render_to(
            scene,
            (20, 80),
            f"Total time: {time_delta:.1f} seconds",
            (0, 0, 0),
            size=16,
        )
    elif isinstance(state.game_over, GameOverCrash):
        media.font.render_to(scene, (20, 20), "Game Over :'(", (0, 0, 0), size=16)

        media.font.render_to(
            scene,
            (20, 40),
            f"Level: {state.level_id + 1} out of {LEVEL_COUNT}",
            (0, 0, 0),
            size=16,
        )

        scene.blit(
            state.game_over.collision.skier_sprite,
            state.game_over.collision.skier_xy,
        )

        scene.blit(
            state.game_over.collision.collider_sprite,
            state.game_over.collision.collider_xy,
        )

        pygame.draw.circle(scene, (255, 0, 0), state.game_over.collision.xy, 5)
    else:
        common.assert_never(state.game_over)

    media.font.render_to(
        scene,
        (20, SCENE_HEIGHT - 20),
        'Press "q" to quit and "r" to restart',
        (0, 0, 0),
        size=10,
    )

    return scene


def render_loading(media: Media) -> pygame.surface.Surface:
    """Render the "Quitting..." dialogue as a scene."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((0, 0, 0))

    media.font.render_to(scene, (20, 20), "Loading...", (255, 255, 255), size=32)

    return scene


def render_quit(media: Media) -> pygame.surface.Surface:
    """Render the "Quitting..." dialogue as a scene."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((0, 0, 0))

    media.font.render_to(scene, (20, 20), "Quitting...", (255, 255, 255), size=32)

    return scene


def resize_image_to_canvas_and_blit(
    image: pygame.surface.Surface, canvas: pygame.surface.Surface
) -> None:
    """Draw the image on canvas resizing it to maximum at constant aspect ratio."""
    canvas.fill((0, 0, 0))

    canvas_aspect_ratio = fractions.Fraction(canvas.get_width(), canvas.get_height())
    image_aspect_ratio = fractions.Fraction(image.get_width(), image.get_height())

    if image_aspect_ratio < canvas_aspect_ratio:
        new_image_height = canvas.get_height()
        new_image_width = image.get_width() * (new_image_height / image.get_height())

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        margin = int((canvas.get_width() - image.get_width()) / 2)

        canvas.blit(image, (margin, 0))

    elif image_aspect_ratio == canvas_aspect_ratio:
        new_image_width = canvas.get_width()
        new_image_height = image.get_height()

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        canvas.blit(image, (0, 0))
    else:
        new_image_width = canvas.get_width()
        new_image_height = int(
            image.get_height() * (new_image_width / image.get_width())
        )

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        margin = int((canvas.get_height() - image.get_height()) / 2)

        canvas.blit(image, (0, margin))


def main(prog: str) -> int:
    """
    Execute the main routine.

    :param prog: name of the program to be displayed in the help
    :return: exit code
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--version", help="show the current version and exit", action="store_true"
    )

    # NOTE (mristin, 2022-12-16):
    # The module ``argparse`` is not flexible enough to understand special options such
    # as ``--version`` so we manually hard-wire.
    if "--version" in sys.argv and "--help" not in sys.argv:
        print(skileu.__version__)
        return 0

    _ = parser.parse_args()

    pygame.init()
    pygame.mixer.pre_init()
    pygame.mixer.init()

    pygame.display.set_caption("Ski Leu")

    surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    print("Loading the media...")
    try:
        media, error = load_media()

        if error is not None:
            print(f"Failed to load the media: {error}", file=sys.stderr)
            return 1

        assert media is not None
    except Exception as exception:
        print(
            f"Failed to load the media: {exception.__class__.__name__} {exception}",
            file=sys.stderr,
        )
        return 1

    print("Showing loading...")
    scene = render_loading(media)
    resize_image_to_canvas_and_blit(scene, surface)
    pygame.display.flip()

    print("Loading the detector...")
    detector = bodypose.load_detector()

    clock = pygame.time.Clock()

    print("Opening the video capture...")
    try:
        cap = cv2.VideoCapture(0)
    except Exception as exception:
        print(f"Failed to open the video capture: {exception}", file=sys.stderr)
        return 1

    try:
        print("Initializing the state...")
        now = pygame.time.get_ticks() / 1000
        level = generate_level(now=now, media=media)
        state = State(game_start=now, media=media, level=level)

        prev_game_over = None  # type: Optional[GameOver]

        print("Entering the endless loop...")
        while cap.isOpened() and not state.received_quit:
            now = pygame.time.get_ticks() / 1000

            reading_ok, frame = cap.read()
            if not reading_ok:
                break

            # Flip so that it is easier to understand the image
            frame = cv2.flip(frame, 1)

            detections = detector(frame)

            frame_with_wire = None  # type: Optional[pygame.surface.Surface]
            if len(detections) > 0:
                detection = detections[0]
                # fmt: on
                (
                    maybe_action,
                    velocity_factor,
                    frame_with_wire,
                ) = recognize_action_from_detection(detection, frame)
                # fmt: off
                if maybe_action is not None:
                    state.skier.action = maybe_action

                state.skier.velocity_factor = velocity_factor

            if frame_with_wire is None:
                frame_with_wire = cvmat_to_surface(frame)

            assert frame_with_wire is not None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state.received_quit = True
                    continue

                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    state.received_quit = True
                    continue

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    level = generate_level(now=now, media=media)
                    state = State(
                        game_start=pygame.time.get_ticks() / 1000,
                        media=media,
                        level=level,
                    )
                    pygame.mixer.music.stop()
                    continue

                else:
                    # Ignore events that we do not handle
                    pass

            update_state_on_tick(state, now, media)

            if state.game_over is not None:
                scene = render_game_over(state, media)
                resize_image_to_canvas_and_blit(scene, surface)
                pygame.display.flip()
            else:
                scene = render_in_game(state, media, frame_with_wire)
                resize_image_to_canvas_and_blit(scene, surface)
                pygame.display.flip()

            if (
                state.game_over is not None
                and prev_game_over is None
                and isinstance(state.game_over, GameOverOk)
            ):
                pygame.mixer.music.load(str(media.success_music_path))
                pygame.mixer.music.play()

            prev_game_over = state.game_over

            # Enforce 30 frames per second
            clock.tick(30)
    finally:
        print("Quitting the game...")
        tic = time.time()

        scene = render_quit(media)
        resize_image_to_canvas_and_blit(scene, surface)
        pygame.display.flip()

        if cap is not None:
            cap.release()

        pygame.quit()
        print(f"Quit the game after: {time.time() - tic:.2f} seconds")

    return 0


def entry_point() -> int:
    """Provide an entry point for a console script."""
    return main(prog="ski-leu")


if __name__ == "__main__":
    sys.exit(main(prog="ski-leu"))
