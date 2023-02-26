"""Recognize the body pose of the player."""

import collections
import enum
from typing import List, Final, Mapping, Callable, MutableMapping

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from icontract import require


class KeypointLabel(enum.Enum):
    """Map keypoints names to the indices in the network output."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


#: Map indices of the network output to keypoint labels
KEYPOINT_INDEX_TO_LABEL = {literal.value: literal for literal in KeypointLabel}


class Keypoint:
    """Represent a single detection of a keypoint in an image."""

    x: float
    y: float
    confidence: float

    @require(lambda confidence: 0 <= confidence <= 1)
    def __init__(self, x: float, y: float, confidence: float) -> None:
        """
        Initialize with the given values.

        :param x: X-coordinate in the image rescaled to [0, 1] x [0, 1]
        :param y: Y-coordinate in the image rescaled to [0, 1] x [0, 1]
        :param confidence: in the range [0,1] of the keypoint detection
        """
        self.x = x
        self.y = y
        self.confidence = confidence


class Detection:
    """Represent a detection of a person in an image."""

    #: Keypoints of the pose
    keypoints: Final[Mapping[KeypointLabel, Keypoint]]

    #: Score of the person detection.
    #:
    #: .. note::
    #:
    #:     This score is the score of the *person* detection, not of the individual
    #:     joints. For the score of the individual joints,
    #:     see :py:attr:`Keypoint.confidence`

    @require(lambda score: 0 <= score <= 1)
    def __init__(
        self,
        keypoints: Mapping[KeypointLabel, Keypoint],
        score: float,
    ) -> None:
        """Initialize with the given values."""
        self.keypoints = keypoints
        self.score = score


def load_detector() -> Callable[[cv2.Mat], List[Detection]]:
    """
    Load the model and return the function which you can readily use on images.

    :return: detector function to be applied on images
    """
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet = model.signatures["serving_default"]

    # If a detection has a score below this threshold, it will be ignored.
    detection_score_threshold = 0.2

    # If a keypoint has a confidence below this threshold, it will be ignored.
    keypoint_confidence_threshold = 0.2

    def apply_model(img: cv2.Mat) -> List[Detection]:
        # NOTE (mristin, 2023-02-26):
        # Vaguely based on:
        # * https://www.tensorflow.org/hub/tutorials/movenet,
        # * https://www.section.io/engineering-education/multi-person-pose-estimator-with-python/,
        # * https://analyticsindiamag.com/how-to-do-pose-estimation-with-movenet/ and
        # * https://github.com/geaxgx/openvino_movenet_multipose/blob/main/MovenetMPOpenvino.py

        # Both height and width need to be multiple of 32,
        # height to width ratio should resemble the original image, and
        # the larger side should be made to 256 pixels.
        #
        # Example: 720x1280 should be resized to 160x256.

        height, width, _ = img.shape

        input_size = 256

        if height > width:
            new_height = input_size
            # fmt: off
            new_width = int(
                (float(width) * float(new_height) / float(height)) // 32
            ) * 32
            # fmt: on
        else:
            new_width = input_size
            # fmt: off
            new_height = int(
                (float(height) * float(new_width) / float(width)) // 32
            ) * 32
            # fmt: on

        if new_height != height or new_width != width:
            resized = cv2.resize(img, (new_width, new_height))
        else:
            resized = img

        tf_input_img = tf.cast(
            tf.image.resize_with_pad(
                image=tf.expand_dims(resized, axis=0),
                target_height=new_height,
                target_width=new_width,
            ),
            dtype=tf.int32,
        )

        inference = movenet(tf_input_img)
        output_as_tensor = inference["output_0"]
        assert output_as_tensor.shape == (1, 6, 56)

        output = np.squeeze(output_as_tensor)
        assert output.shape == (6, 56)

        detections = []  # type: List[Detection]

        for detection_i in range(6):
            kps = output[detection_i][:51].reshape(17, -1)
            bbox = output[detection_i][51:55].reshape(2, 2)
            score = output[detection_i][55]

            if score < detection_score_threshold:
                continue

            assert kps.shape == (17, 3)
            assert bbox.shape == (2, 2)

            kps_xy = kps[:, [1, 0]]
            kps_confidence = kps[:, 2]

            assert kps_xy.shape == (17, 2)
            assert kps_confidence.shape == (17,)

            keypoints = (
                collections.OrderedDict()
            )  # type: MutableMapping[KeypointLabel, Keypoint]

            for keypoint_i in range(17):
                label = KEYPOINT_INDEX_TO_LABEL[keypoint_i]
                kp_x, kp_y = kps_xy[keypoint_i, :]
                kp_confidence = kps_confidence[keypoint_i]

                if kp_confidence < keypoint_confidence_threshold:
                    continue

                assert label not in keypoints
                keypoints[label] = Keypoint(kp_x, kp_y, kp_confidence)

            detection = Detection(keypoints, score)

            detections.append(detection)

        return detections

    return apply_model
