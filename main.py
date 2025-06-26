# TERMINOLOGY:
# - junction: a point where 3 or more lines meet
# - termination: a point where 1 line ends
# - node: either a junction or a termination
# - line: A connection between two nodes
# - endpoint: A point on either end of a line which needs to be associated with a node

import config
import cv2 as cv
import math
import numpy as np
from enum import Enum

# import hardware

print("\r\n\r\nDRC Main Launch File")

print(config.values)

# hardware_api = hardware.HardwareAPI()

window_title = "Pathfinder"

img = cv.imread("paths.png")
rows, cols, channels = img.shape
rows = rows + 2
cols = cols + 2

cv.namedWindow(window_title, cv.WINDOW_GUI_NORMAL)
cv.resizeWindow(window_title, int(cols), int(rows))

draw_junctions = False
draw_terminations = False


class ShownImage(Enum):
    INPUT = 0
    SKELETON = 1
    JUNCTIONS = 2
    FILTERED_JUNCTIONS = 3
    TERMINATIONS = 4
    FILTERED_TERMINATIONS = 5


shown_image = ShownImage.INPUT
while True:
    key = cv.waitKey(1)
    if key == ord("-"):
        break
    if key == ord("a"):
        draw_junctions = not draw_junctions
    if key == ord("s"):
        draw_terminations = not draw_terminations
    if key == ord("1"):
        shown_image = ShownImage.INPUT
    if key == ord("2"):
        shown_image = ShownImage.SKELETON
    if key == ord("3"):
        shown_image = ShownImage.JUNCTIONS
    if key == ord("4"):
        shown_image = ShownImage.FILTERED_JUNCTIONS
    if key == ord("5"):
        shown_image = ShownImage.TERMINATIONS
    if key == ord("6"):
        shown_image = ShownImage.FILTERED_TERMINATIONS

    img = cv.imread("paths.png")
    # create black border to prevent weird skeletonization artifacts
    img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    skeleton = cv.ximgproc.thinning(img_gray, thinningType=cv.ximgproc.THINNING_GUOHALL)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel = kernel / np.sum(kernel)
    blur = cv.filter2D(skeleton, -1, kernel)
    # averaging blur
    _, junctions_raw = cv.threshold(blur, 110, 255, cv.THRESH_BINARY)
    _, terminations_raw = cv.threshold(blur, 80, 255, cv.THRESH_BINARY_INV)
    terminations_raw = cv.bitwise_and(terminations_raw, skeleton)
    junctions_raw = cv.bitwise_and(junctions_raw, skeleton)

    terminations = cv.dilate(terminations_raw, kernel, iterations=1)
    junctions = cv.dilate(junctions_raw, kernel, iterations=1)

    # with junctions found:
    # - strip them out of the skeletonised image
    # - calculate contours
    # - draw each contour individually on black frame
    # - find endpoints
    # - connect endpoints to nearest 2 junctions
    stripped_skel = skeleton.copy()
    stripped_skel[junctions > 0] = 0
    line_contours, heierarchy = cv.findContours(
        stripped_skel, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    junctions = cv.dilate(junctions, kernel, iterations=1)
    terminations = cv.dilate(terminations, kernel, iterations=1)

    nodes = cv.bitwise_or(junctions, terminations)

    endpoints = cv.bitwise_and(stripped_skel, nodes)

    endpoint_coords = np.argwhere(endpoints).astype(np.int32)
    # print(endpoint_coords)

    junction_contours, _ = cv.findContours(
        junctions, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    termination_contours, _ = cv.findContours(
        terminations, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    node_indices = range(len(junction_contours) + len(termination_contours))
    line_indices = range(len(line_contours))
    endpoint_indices = range(len(endpoint_coords))

    node_map = {}

    links = {}

    # associate endpoints with junctions
    for endpoint_idx in endpoint_indices:
        endpoint = endpoint_coords[endpoint_idx]
        for node_idx in node_indices:
            contour = (
                junction_contours[node_idx]
                if node_idx < len(junction_contours)
                else termination_contours[node_idx - len(junction_contours)]
            )
            dist = cv.pointPolygonTest(
                contour, (int(endpoint[1]), int(endpoint[0])), False
            )
            if dist < 0:
                continue
            if node_idx not in node_map:
                node_map[node_idx] = []
            node_map[node_idx].append(endpoint_idx)
            break
    print("Node map:", node_map)

    # 0.15: too low
    # 0.2: endpoints, junctions
    # 0.4: most junctions (shallow ones fail)
    # corners = cv.goodFeaturesToTrack(skeleton, 25, 0.2, 10)

    disp = None
    match shown_image:
        case ShownImage.INPUT:
            disp = img.copy()
        case ShownImage.SKELETON:
            disp = cv.cvtColor(skeleton, cv.COLOR_GRAY2BGR)
        case ShownImage.JUNCTIONS:
            disp = cv.cvtColor(junctions_raw, cv.COLOR_GRAY2BGR)
        case ShownImage.FILTERED_JUNCTIONS:
            disp = cv.cvtColor(junctions, cv.COLOR_GRAY2BGR)
        case ShownImage.TERMINATIONS:
            disp = cv.cvtColor(terminations_raw, cv.COLOR_GRAY2BGR)
        case ShownImage.FILTERED_TERMINATIONS:
            disp = cv.cvtColor(terminations, cv.COLOR_GRAY2BGR)

    if draw_junctions:
        disp[junctions > 0] = (0, 255, 0)
    if draw_terminations:
        disp[terminations > 0] = (255, 0, 255)  # endpoints

    for junction in line_contours:
        # cv.drawContours(disp, [contour], 0, (0, 0, 255), 3)
        x, y, w, h = cv.boundingRect(junction)
        # cv.rectangle(disp, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
