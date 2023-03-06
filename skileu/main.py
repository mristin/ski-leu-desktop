"""Ski through a dangerous forest."""

import argparse
import enum
import fractions
import importlib
import os
import pathlib
import random
import sys
import time
from typing import List, Tuple, Optional, Union

import cv2
import pygame
import pygame.freetype
from icontract import require, ensure

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
        self.margin_sprites_mirrored = [
            pygame.transform.flip(sprite, True, False) for sprite in margin_sprites
        ]

        self.font = font

        self.mask_map = {
            sprite: pygame.mask.from_surface(sprite)
            for sprite in [
                self.skier_forward_sprite,
                self.skier_left_sprite,
                self.skier_right_sprite,
            ]
            + self.obstacle_sprites
            + self.margin_sprites
            + self.margin_sprites_mirrored
        }


SCENE_WIDTH = 800
SCENE_HEIGHT = 600

ROAD_MARGIN = 128


class ActorSpriteSet:
    """Represent actor sprites."""

    #: Sprites of the actor idling around
    idle: List[pygame.surface.Surface]

    #: Sprites of the actor walking to the side
    walk: List[pygame.surface.Surface]

    def __init__(
        self, idle: List[pygame.surface.Surface], walk: List[pygame.surface.Surface]
    ) -> None:
        """Initialize with the given values."""
        self.idle = idle
        self.walk = walk


# TODO (mristin, 2023-03-5): load and crop at loading — supply parameter sprite_count in the cropping function
# TODO (mristin, 2023-03-5): action_start — timestamp when the action started
# TODO (mristin, 2023-03-5): velocity: float 🠒 < 0 left, > 0 right, 0 idle
# TODO (mristin, 2023-03-5): action_eta — action_start + random(2, 3) seconds
# TODO (mristin, 2023-03-5):   pick random direction or idle 1/3, 1/3, 1/3,
# TODO (mristin, 2023-03-5):   if too close to boundary: pick only idle or move to other direction
# TODO (mristin, 2023-03-5): if would hit the boundary, go to idle, next_action = now

# TODO (mristin, 2023-03-5): every row: 1/2, 1/2 trees or actor

# TODO (mristin, 2023-03-5): https://pygame.readthedocs.io/en/latest/tiles/tiles.html
#          self.tiles = []
#         x0 = y0 = self.margin
#         w, h = self.rect.size
#         dx = self.size[0] + self.spacing
#         dy = self.size[1] + self.spacing
#
#         for x in range(x0, w, dx):
#             for y in range(y0, h, dy):
#                 tile = pygame.Surface(self.size)
#                 tile.blit(self.image, (0, 0), (x, y, *self.size))
#                 self.tiles.append(tile)


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

    return (
        Media(
            skier_forward_sprite=skier_forward_sprite,
            skier_left_sprite=skier_left_sprite,
            skier_right_sprite=skier_right_sprite,
            obstacle_sprites=obstacle_sprites,
            margin_sprites=margin_sprites,
            font=font,
        ),
        None,
    )


class Obstacle:
    """Represent an obstacle in the level."""

    #: Appearance
    sprite: pygame.surface.Surface

    #: Top-left corner, in world coordinates
    xy: Tuple[int, int]

    def __init__(self, sprite: pygame.surface.Surface, xy: Tuple[int, int]) -> None:
        """Initialize with the given values."""
        self.sprite = sprite
        self.xy = xy


def calculate_obstacle_bounding_box(obstacle: Obstacle) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box of the obstacle based on its location.

    In the world coordinates, (xmin, ymin, xmax, ymax).
    """
    return (
        obstacle.xy[0],
        obstacle.xy[1],
        obstacle.xy[0] + obstacle.sprite.get_width(),
        obstacle.xy[1] + obstacle.sprite.get_height(),
    )


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
            media.skier_right_sprite,
        ]
    )


def calculate_skier_height(media: Media) -> int:
    """Calculate the skier width based on the sprites."""
    return max(
        sprite.get_height()
        for sprite in [
            media.skier_forward_sprite,
            media.skier_left_sprite,
            media.skier_right_sprite,
        ]
    )


def generate_level(media: Media) -> Level:
    """Generate randomly a level."""
    obstacles = []  # type: List[Obstacle]

    assert all(sprite.get_height() > 0 for sprite in media.margin_sprites), (
        "All margin sprites at least 1 pixel tall "
        "so that we do not enter an endless loop"
    )

    # region Generate left margin
    cursor = 0
    while cursor < SCENE_HEIGHT:
        sprite = random.choice(media.margin_sprites)
        obstacles.append(Obstacle(sprite=sprite, xy=(0, cursor)))
        cursor += sprite.get_height()
    # endregion

    # region Generate right margin
    cursor = 0
    while cursor < SCENE_HEIGHT:
        sprite = random.choice(media.margin_sprites_mirrored)
        obstacles.append(
            Obstacle(sprite=sprite, xy=(SCENE_WIDTH - sprite.get_width(), cursor))
        )
        cursor += sprite.get_height()
    # endregion

    # region Generate obstacles
    assert all(sprite.get_height() > 0 for sprite in media.obstacle_sprites), (
        "All obstacle sprites at least 1 pixel tall "
        "so that we do not enter an endless loop"
    )

    max_obstacle_height = max(sprite.get_height() for sprite in media.obstacle_sprites)

    row_height = round(max_obstacle_height * 2.5)
    padding = round((row_height - max_obstacle_height) / 2)

    skier_width = calculate_skier_width(media)

    margin_width = max(sprite.get_width() for sprite in media.margin_sprites)

    # Skp the first row so that the player can get prepared
    cursor = 2 * calculate_skier_height(media)
    while cursor < SCENE_HEIGHT - max_obstacle_height:
        last_x = None  # type: Optional[int]
        while True:
            obstacle_sprite = random.choice(media.obstacle_sprites)

            if last_x is None:
                x = margin_width + random.randint(0, skier_width)
            else:
                x = last_x + random.randint(2 * skier_width, 3 * skier_width)

            last_x = x

            # Add some jitter to y to make the level more natural
            y = cursor + obstacle_sprite.get_height() + random.randint(0, padding)

            # We can not display this obstacle, so we are done for
            # the row.
            if x + obstacle_sprite.get_width() >= SCENE_WIDTH - margin_width:
                break

            obstacles.append(Obstacle(sprite=obstacle_sprite, xy=(x, y)))

        cursor += row_height
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
    center_xy: Tuple[int, int]

    action: SkierAction

    def __init__(self, center_xy: Tuple[int, int], action: SkierAction) -> None:
        """Initialize with the given values."""
        self.center_xy = center_xy
        self.action = action


def skier_action_to_sprite(action: SkierAction, media: Media) -> pygame.surface.Surface:
    """Determine the sprite corresponding to the skier action."""
    if action is SkierAction.LEFT:
        return media.skier_left_sprite
    elif action is SkierAction.RIGHT:
        return media.skier_right_sprite
    elif action is SkierAction.FORWARD:
        return media.skier_forward_sprite
    else:
        common.assert_never(action)


def calculate_skier_xmin_ymin(
    xy: Tuple[int, int], skier_sprite: pygame.surface.Surface
) -> Tuple[int, int]:
    """
    Calculate the skier corner.

    In world coordinates.
    """
    return (
        xy[0] - skier_sprite.get_width() // 2,
        xy[1] - skier_sprite.get_height() // 2,
    )


def calculate_skier_bounding_box(
    center_xy: Tuple[int, int], skier_sprite: pygame.surface.Surface
) -> Tuple[int, int, int, int]:
    """
    Compute (xmin, ymin, xmax, ymax) based on the center (x, y).

    In world coordinates.
    """
    xmin, ymin = calculate_skier_xmin_ymin(center_xy, skier_sprite)

    xmax = xmin + skier_sprite.get_width()
    ymax = ymin + skier_sprite.get_height()

    return xmin, ymin, xmax, ymax


class GameOver:
    """Capture how and when the game ended."""

    timestamp: float

    def __init__(self, timestamp: float) -> None:
        """Initialize with the given values."""
        self.timestamp = timestamp


class GameOverOk(GameOver):
    """Represent a successful game."""


class GameOverCrash(GameOver):
    #: Obstacle that the skier had a collision with
    obstacle: Obstacle

    #: Collision location, in world coordinates
    collision_xy: Tuple[int, int]

    def __init__(
        self, timestamp: float, obstacle: Obstacle, collision_xy: Tuple[int, int]
    ) -> None:
        """Initialize with the given values."""
        GameOver.__init__(self, timestamp)
        self.obstacle = obstacle
        self.collision_xy = collision_xy


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

    skier_height = calculate_skier_height(media)
    state.skier = Skier(
        center_xy=(round(SCENE_WIDTH / 2), -skier_height), action=SkierAction.FORWARD
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


VELOCITY_FACTOR = 0.5

#: Velocity in world coordinates depending on action, (x, y)
VELOCITY_DISPATCH = {
    SkierAction.FORWARD: (0, VELOCITY_FACTOR * 50),
    SkierAction.LEFT: (-50, VELOCITY_FACTOR * 30),
    SkierAction.RIGHT: (50, VELOCITY_FACTOR * 30),
}
assert all(action in VELOCITY_DISPATCH for action in SkierAction)


def update_state_on_tick(state: State, now: float, media: Media) -> None:
    """Update state on one game cycle."""
    time_delta = now - state.now

    state.now = now

    if state.game_over is not None:
        return

    # region Check for collision(s) between obstacles and the skier
    skier_sprite = skier_action_to_sprite(state.skier.action, media)

    skier_bbox = calculate_skier_bounding_box(state.skier.center_xy, skier_sprite)

    for obstacle in state.level.obstacles:
        obstacle_bbox = calculate_obstacle_bounding_box(obstacle)

        if intersect(
            skier_bbox[0],
            skier_bbox[1],
            skier_bbox[2],
            skier_bbox[3],
            obstacle_bbox[0],
            obstacle_bbox[1],
            obstacle_bbox[2],
            obstacle_bbox[3],
        ):
            skier_mask = media.mask_map[skier_sprite]
            obstacle_mask = media.mask_map[obstacle.sprite]

            # NOTE (mristin, 2023-03-05):
            # World coordinates start in bottom-left corner. Screen coordinates start in
            # top-left.
            #
            # The mask offsets need to be computed in the screen coordinates.
            # See: https://www.pygame.org/docs/ref/mask.html#mask-offset-label

            skier_world_xmin, _, _, skier_world_ymax = skier_bbox
            skier_screen_xy = world_xy_to_screen_xy(
                (skier_world_xmin, skier_world_ymax)
            )

            obstacle_world_xmin, _, _, obstacle_world_ymax = obstacle_bbox
            obstacle_screen_xy = world_xy_to_screen_xy(
                (obstacle_world_xmin, obstacle_world_ymax)
            )

            offset = (
                obstacle_screen_xy[0] - skier_screen_xy[0],
                obstacle_screen_xy[1] - skier_screen_xy[1],
            )

            collision_screen_xy = skier_mask.overlap(obstacle_mask, offset)

            if collision_screen_xy is not None:
                state.game_over = GameOverCrash(
                    timestamp=now,
                    obstacle=obstacle,
                    collision_xy=(
                        collision_screen_xy[0] + skier_screen_xy[0],
                        # NOTE (mristin, 2023-03-05):
                        # We convert here screen to world coordinates.
                        SCENE_HEIGHT - (collision_screen_xy[1] + skier_screen_xy[1]),
                    ),
                )
                return
    # endregion

    # region Check for reaching the end of level
    skier_ymax = skier_bbox[3]
    if skier_ymax >= SCENE_HEIGHT:
        if state.level_id == LEVEL_COUNT - 1:
            state.game_over = GameOverOk(now)
        else:
            state.level_id += 1
            state.level = generate_level(media=media)

            skier_height = calculate_skier_height(media)
            state.skier.center_xy = (state.skier.center_xy[0], -skier_height)

        return
    # endregion

    # region Update skier
    velocity = VELOCITY_DISPATCH[state.skier.action]

    state.skier.center_xy = (
        state.skier.center_xy[0] + velocity[0] * time_delta,
        state.skier.center_xy[1] + velocity[1] * time_delta,
    )
    # endregion


def cvmat_to_surface(image: cv2.Mat) -> pygame.surface.Surface:
    """Convert from OpenCV to pygame."""
    height, width, _ = image.shape
    if height == 0 and width == 0:
        return pygame.surface.Surface((1, 1))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image_rgb.tobytes(), (width, height), "RGB")


def action_from_detection(
    detection: bodypose.Detection, frame: cv2.Mat
) -> Tuple[Optional[SkierAction], pygame.surface.Surface]:
    """
    Infer the action based on the body pose detection.

    Return the action and the sprite showing it.
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

    return action, cvmat_to_surface(frame_with_wire)


def world_xy_to_screen_xy(xy: Tuple[int, int]) -> Tuple[int, int]:
    """Convert the world coordinates to screen coordinates, as (x,y)."""
    return xy[0], SCENE_HEIGHT - xy[1]


def draw_obstacle_on_scene(scene: pygame.surface.Surface, obstacle: Obstacle) -> None:
    """Draw the obstacle on the scene."""
    # NOTE (mristin, 2023-03-05):
    # World coordinates start in bottom-left corner. Screen coordinates start in
    # top-left.
    obstacle_bbox = calculate_obstacle_bounding_box(obstacle)
    obstacle_world_xmin, _, _, obstacle_world_ymax = obstacle_bbox

    obstacle_screen_xy = world_xy_to_screen_xy(
        (obstacle_world_xmin, obstacle_world_ymax)
    )

    scene.blit(obstacle.sprite, obstacle_screen_xy)


def draw_skier_on_scene(
    scene: pygame.surface.Surface, skier: Skier, media: Media
) -> None:
    """Draw the skier on the scene."""
    skier_sprite = skier_action_to_sprite(skier.action, media)

    # NOTE (mristin, 2023-03-05):
    # World coordinates start in bottom-left corner. Screen coordinates start in
    # top-left.
    skier_bbox = calculate_skier_bounding_box(skier.center_xy, skier_sprite)
    skier_world_xmin, _, _, skier_world_ymax = skier_bbox

    skier_screen_xy = world_xy_to_screen_xy((skier_world_xmin, skier_world_ymax))

    scene.blit(skier_sprite, skier_screen_xy)


@require(lambda state: state.game_over is None)
def render_in_game(
    state: State, media: Media, frame_with_wire: pygame.surface.Surface
) -> pygame.surface.Surface:
    """Render the game screen based on the state."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((255, 255, 255))

    for obstacle in state.level.obstacles:
        draw_obstacle_on_scene(scene, obstacle)

    draw_skier_on_scene(scene, state.skier, media)

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
    elif isinstance(state.game_over, GameOverCrash):
        media.font.render_to(scene, (20, 20), "Game Over :'(", (0, 0, 0), size=16)

        media.font.render_to(
            scene,
            (20, 40),
            f"Level: {state.level_id + 1} out of {LEVEL_COUNT}",
            (0, 0, 0),
            size=16,
        )

        draw_skier_on_scene(scene, state.skier, media)

        draw_obstacle_on_scene(scene, state.game_over.obstacle)

        pygame.draw.circle(
            scene, (255, 0, 0), world_xy_to_screen_xy(state.game_over.collision_xy), 5
        )
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
        level = generate_level(media)
        now = pygame.time.get_ticks() / 1000
        state = State(game_start=now, media=media, level=level)

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
                maybe_action, frame_with_wire = action_from_detection(detection, frame)
                if maybe_action is not None:
                    state.skier.action = maybe_action

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
                    level = generate_level(media)
                    state = State(
                        game_start=pygame.time.get_ticks() / 1000,
                        media=media,
                        level=level,
                    )
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
