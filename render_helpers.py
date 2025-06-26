import cv2 as cv
import config
import math


def render_text(img, text, origin, color=(0, 0, 0), border=(255, 255, 255), scale=1):
    processing_scale = config.values["algorithm"]["processing_scale"]
    scale = scale * processing_scale
    (x, y) = origin
    origin = (int(x * processing_scale), int(y * processing_scale))
    cv.putText(
        img,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        border,
        math.ceil(3 * scale),
        cv.LINE_AA,
    )
    cv.putText(
        img,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        math.ceil(1 * scale),
        cv.LINE_AA,
    )
