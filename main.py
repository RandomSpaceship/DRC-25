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
    BLUR = 6
    SKELETON_MINUS_JUNCTIONS = 7
    ENDPOINTS = 8


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
    if key == ord("7"):
        shown_image = ShownImage.BLUR
    if key == ord("8"):
        shown_image = ShownImage.SKELETON_MINUS_JUNCTIONS
    if key == ord("9"):
        shown_image = ShownImage.ENDPOINTS

    img = cv.imread("paths2.png")
    # create black border to prevent weird skeletonization artifacts
    img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    skeleton = cv.ximgproc.thinning(img_gray, thinningType=cv.ximgproc.THINNING_GUOHALL)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel = kernel / np.sum(kernel)
    # averaging blur
    blur = cv.filter2D(skeleton, -1, kernel)
    junction_thresh = int(255 * 3.5 / 9)
    termination_thresh = int(255 * 2.5 / 9)
    _, junctions_raw = cv.threshold(blur, junction_thresh, 255, cv.THRESH_BINARY)
    _, terminations_raw = cv.threshold(
        blur, termination_thresh, 255, cv.THRESH_BINARY_INV
    )
    terminations_raw = cv.bitwise_and(terminations_raw, skeleton)
    junctions_raw = cv.bitwise_and(junctions_raw, skeleton)

    # terminations we can guarantee are single pixels only, junctions can be larger with specific geometry
    terminations = terminations_raw

    # with junctions found:
    # - strip them out of the skeletonised image
    # - calculate contours
    # - draw each contour individually on black frame
    # - find endpoints
    # - connect endpoints to nearest 2 junctions
    stripped_skel = skeleton.copy()
    stripped_skel[junctions_raw > 0] = 0
    line_contours, _ = cv.findContours(
        stripped_skel, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    junctions = cv.dilate(junctions_raw, kernel, iterations=1)

    junction_endpoints = cv.bitwise_and(stripped_skel, junctions)

    junction_endpoint_coords = np.argwhere(junction_endpoints)
    termination_coords = np.argwhere(terminations)
    endpoint_coords = np.concatenate(
        (junction_endpoint_coords, termination_coords), axis=0
    )

    junction_contours, _ = cv.findContours(
        junctions, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    junction_endpoint_count = len(junction_endpoint_coords)
    termination_count = len(termination_coords)
    endpoint_count = junction_endpoint_count + termination_count

    junction_count = len(junction_contours)
    node_count = junction_count + termination_count

    line_count = len(line_contours)

    junction_endpoints = [[] for _ in range(junction_count)]
    line_endpoints = [[] for _ in range(line_count)]

    # associate junction endpoints with their junction
    for endpoint_idx, endpoint in enumerate(junction_endpoint_coords):
        for junction_idx, contour in enumerate(junction_contours):
            dist = cv.pointPolygonTest(
                contour, (int(endpoint[1]), int(endpoint[0])), False
            )
            if dist < 0:
                continue
            junction_endpoints[junction_idx].append(endpoint_idx)
            break

    # associate endpoints with their lines
    for line_idx, contour in enumerate(line_contours):
        for endpoint_idx, endpoint in enumerate(
            np.vstack((junction_endpoint_coords, termination_coords))
        ):
            dist = cv.pointPolygonTest(
                contour, (int(endpoint[1]), int(endpoint[0])), False
            )
            if dist >= 0:
                line_endpoints[line_idx].append(endpoint_idx)

    tree_roots = []
    for current_idx, (y, x) in enumerate(termination_coords):
        if y >= rows - 2:
            tree_roots.append(current_idx + junction_endpoint_count)

    def get_contour_center(contour):
        """Calculate the center of a contour."""
        M = cv.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    junction_coords = [get_contour_center(contour) for contour in junction_contours]
    line_lengths = [cv.arcLength(contour, True) // 2 for contour in line_contours]

    def find_junction(endpoint_idx):
        """Find the junction index for a given endpoint index."""
        for junction_idx, endpoints in enumerate(junction_endpoints):
            if endpoint_idx in endpoints:
                tmp = endpoints.copy()
                tmp.remove(endpoint_idx)
                return (junction_idx, tmp)
        return None

    def find_line(endpoint_idx):
        """Find the line index for a given endpoint index."""
        for line_idx, endpoints in enumerate(line_endpoints):
            if endpoint_idx in endpoints:
                if endpoints[0] == endpoint_idx:
                    return (line_idx, endpoints[1])
                else:
                    return (line_idx, endpoints[0])
        return None

    tree = {}
    for tree_idx, root_idx in enumerate(tree_roots):
        leaf_indices = [(root_idx, root_idx)]
        while leaf_indices:
            (prev_idx, current_node_idx) = leaf_indices.pop(0)
            if prev_idx not in tree:
                tree[prev_idx] = []

            line_data = find_line(current_node_idx)
            if line_data is None:
                print("Error: No line found for endpoint", current_node_idx)
                continue
            line_idx, other_endpoint_idx = line_data

            # if it's a junction endpoint...
            if other_endpoint_idx < junction_endpoint_count:
                junction_data = find_junction(other_endpoint_idx)
                if junction_data is None:
                    print("Error: No junction found for endpoint", current_node_idx)
                    continue
                junction_idx, other_nodes = junction_data
                tree[prev_idx].append((junction_idx, line_idx))
                leaf_indices.extend([(junction_idx, idx) for idx in other_nodes])
            else:
                # this is a termination
                tree[prev_idx].append((other_endpoint_idx, line_idx))

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
        case ShownImage.BLUR:
            disp = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)
        case ShownImage.SKELETON_MINUS_JUNCTIONS:
            disp = cv.cvtColor(stripped_skel, cv.COLOR_GRAY2BGR)
        case ShownImage.ENDPOINTS:
            disp = cv.cvtColor(junction_endpoints, cv.COLOR_GRAY2BGR)

    to_draw = tree_roots.copy()
    while to_draw:
        current_idx = to_draw.pop(0)
        if current_idx in tree:
            to_draw.extend(tree[current_idx][0])
        if current_idx >= 0 and current_idx in tree:
            current_pos = (
                junction_coords[current_idx]
                if current_idx < len(junction_coords)
                else (
                    termination_coords[current_idx - junction_endpoint_count][1],
                    termination_coords[current_idx - junction_endpoint_count][0],
                )
            )
            for child_idx, line_idx in tree[current_idx]:
                child_pos = (
                    junction_coords[child_idx]
                    if child_idx < len(junction_coords)
                    else (
                        termination_coords[child_idx - junction_endpoint_count][1],
                        termination_coords[child_idx - junction_endpoint_count][0],
                    )
                )
                cv.arrowedLine(
                    disp,
                    current_pos,
                    child_pos,
                    (255, 255, 0),
                    1,
                )
                cv.putText(
                    disp,
                    str(line_lengths[line_idx]),
                    (
                        (current_pos[0] + child_pos[0]) // 2,
                        (current_pos[1] + child_pos[1]) // 2,
                    ),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                )

    if draw_junctions:
        disp[junctions > 0] = (0, 255, 0)
    if draw_terminations:
        disp[terminations > 0] = (255, 0, 255)  # endpoints

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
