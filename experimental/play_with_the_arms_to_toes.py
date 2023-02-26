"""Play with the detection of the distance between the arms and the toes."""

import argparse
import math
import sys
from typing import Tuple, Optional

import cv2
from icontract import require

import skileu.bodypose


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

            detections = detector(frame)

            if len(detections) > 0:
                # NOTE (mristin, 2023-02-26):
                # Simply assume the first detection. This can be annoying, but ok for
                # the state of the experiment.
                detection = detections[0]

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

                left_wrist = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.LEFT_WRIST, None
                )

                right_wrist = detection.keypoints.get(
                    skileu.bodypose.KeypointLabel.RIGHT_WRIST, None
                )

                if (
                    left_knee is not None
                    and right_knee is not None
                    and left_ankle is not None
                    and right_ankle is not None
                    and left_wrist is not None
                    and right_wrist is not None
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

                    cv2.line(frame, knee, ankle, (255, 255, 255), 2)
                    cv2.circle(frame, knee, 4, (0, 255, 0), -1)
                    cv2.circle(frame, ankle, 4, (0, 0, 255), -1)

                    cv2.circle(frame, wrist, 4, (255, 0, 0), -1)

                    # NOTE (mristin, 2023-03-08):
                    # The screen coordinates start in the top-left corner.
                    ratio = min(
                        1.0, max(0.0, (ankle[1] - wrist[1]) / (ankle[1] - knee[1]))
                    )

                    # NOTE (mristin, 2023-03-08):
                    # Ratio of 20 seems to make sense.

                    cv2.putText(
                        frame,
                        f"{(ratio * 100):.2f}",
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        3,
                    )

            cv2.imshow(__name__, frame)
            if cv2.waitKey(1) == ord("q"):
                should_stop = True

    finally:
        if cap is not None:
            cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
