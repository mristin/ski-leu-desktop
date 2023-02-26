"""Play with the detection of the angle defined by the knees."""

import argparse
import math
import sys
from typing import Tuple, Optional

import cv2
from icontract import require

import skileu.bodypose


@require(
    lambda hip, knee, ankle: hip[1] > ankle[1] and knee[1] > ankle[1],
    "Coordinate origin in the bottom-left of the image, not in the top-left",
)
def compute_knee_angle(
    hip: Tuple[float, float], knee: Tuple[float, float], ankle: Tuple[float, float]
) -> float:
    """
    Compute the angle between the knee and the other two points.

    Going right means the negative angle:
    >>> round(compute_knee_angle((0, 2), (1, 1), (0, 0)))
    -90

    Squatting means smaller angle when going to the right:
    >>> round(compute_knee_angle((0, 0.5), (1, 0.25), (0, 0)))
    -28

    Going left means the positive angle:
    >>> round(compute_knee_angle((0, 2), (-1, 1), (0, 0)))
    90

    Squatting means smaller angle also to the left:
    >>> round(compute_knee_angle((0, 0.5), (-1, 0.25), (0, 0)))
    28

    Going straight means 180:
    >>> round(compute_knee_angle((0, 2), (0, 1), (0, 0)))
    180

    Some observations regarding the body pose:

    * It seems that going left/right is indicated when the angle goes below 150 degrees.
    * The bent of 120 degrees is already a stretch for the body.
    * 90 degrees is almost impossible.
    """
    # See: https://stackoverflow.com/a/31334882/1600678
    rads = math.atan2(hip[1] - knee[1], hip[0] - knee[0]) - math.atan2(
        ankle[1] - knee[1], ankle[0] - knee[0]
    )

    degrees = rads / math.pi * 180.0
    if degrees > 180:
        # NOTE (mristin, 2023-02-26):
        # We transform the angle so that we can simply compare the sign for
        # the direction and the magnitude for the speed.
        degrees = -(360 - degrees)

    return degrees


def main() -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()

    detector = skileu.bodypose.load_detector()

    cap = cv2.VideoCapture(0)

    should_stop = False

    position = None  # type: Optional[Tuple[float, float]]

    try:
        while cap.isOpened() and not should_stop:
            reading_ok, frame = cap.read()
            if not reading_ok:
                break

            # Flip so that it is easier to understand the image
            frame = cv2.flip(frame, 1)

            frame_height, frame_width, _ = frame.shape

            if position is None:
                position = (
                    int(round(frame_width / 2.0)),
                    int(round(frame_height / 2.0)),
                )

            detections = detector(frame)

            if len(detections) > 0:
                # NOTE (mristin, 2023-02-26):
                # Simply assume the first detection. This can be annoying, but ok for
                # the state of the experiment.
                detection = detections[0]

                left_hip = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.LEFT_HIP, None
                )

                right_hip = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.RIGHT_HIP, None
                )

                left_knee = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.LEFT_KNEE, None
                )

                right_knee = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.RIGHT_KNEE, None
                )

                left_ankle = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.LEFT_ANKLE, None
                )

                right_ankle = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.RIGHT_ANKLE, None
                )

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

                    cv2.line(frame, hip, knee, (255, 255, 255), 2)
                    cv2.line(frame, knee, ankle, (255, 255, 255), 2)

                    cv2.circle(frame, hip, 4, (255, 0, 0), -1)
                    cv2.circle(frame, knee, 4, (0, 255, 0), -1)
                    cv2.circle(frame, ankle, 4, (0, 0, 255), -1)

                    if hip[1] < ankle[1] and knee[1] < ankle[1]:
                        angle = compute_knee_angle(
                            (hip[0], frame_height - hip[1]),
                            (knee[0], frame_height - knee[1]),
                            (ankle[0], frame_height - ankle[1]),
                        )

                        cv2.putText(
                            frame,
                            f"{angle:.2f}",
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            3,
                        )

                        if abs(angle) < 150:
                            direction = 1 if angle < 0 else -1

                            sidewards_velocity = 10
                            position = (
                                max(
                                    0,
                                    min(
                                        frame_width,
                                        position[0] + direction * sidewards_velocity,
                                    ),
                                ),
                                position[1],
                            )
                        else:
                            total = ankle[1] - hip[1]
                            thigh = knee[1] - hip[1]
                            lower_leg = ankle[1] - knee[1]

                            thigh_percentage = round(thigh / total * 100)
                            lower_leg_percentage = round(lower_leg / total * 100)

                            cv2.rectangle(
                                frame,
                                (0, 0),
                                (round(thigh / total * frame_width), 20),
                                (0, 0, 255),
                                3,
                            )
                            cv2.rectangle(
                                frame,
                                (0, 20),
                                (round(lower_leg / total * frame_width), 40),
                                (255, 0, 0),
                                3,
                            )

                            cv2.putText(
                                frame,
                                f"thigh {thigh_percentage}%, "
                                f"lower leg {lower_leg_percentage}%",
                                (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                3,
                            )

            cv2.circle(frame, position, 30, (255, 255, 255), -1)

            cv2.imshow(__name__, frame)
            if cv2.waitKey(1) == ord("q"):
                should_stop = True

    finally:
        if cap is not None:
            cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
