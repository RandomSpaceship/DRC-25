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
import pathfinder

# import hardware

print("\r\n\r\nDRC Main Launch File")

print(config.values)

# hardware_api = hardware.HardwareAPI()

window_title = "Pathfinder"

cv.namedWindow(window_title, cv.WINDOW_GUI_NORMAL)

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

    img = cv.imread("paths7.png")
    input_mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    path_data = pathfinder.find_paths(input_mask)
    tree = path_data["tree"]
    tree_roots = path_data["roots"]
    junction_coords = path_data["junctions"]
    junction_endpoint_count = path_data["junction_endpoint_count"]
    termination_coords = path_data["terminations"]
    line_lengths = path_data["line_lengths"]
    pathfinding_time = path_data["proc_time"]
    print(f"Pathfinding took {pathfinding_time:.3f} seconds")

    disp = img.copy()
    # match shown_image:
    #     case ShownImage.INPUT:
    #         disp = img.copy()
    #     case ShownImage.SKELETON:
    #         disp = cv.cvtColor(skeleton, cv.COLOR_GRAY2BGR)
    #     case ShownImage.JUNCTIONS:
    #         disp = cv.cvtColor(junctions, cv.COLOR_GRAY2BGR)
    #     case ShownImage.FILTERED_JUNCTIONS:
    #         disp = cv.cvtColor(expanded_junctions, cv.COLOR_GRAY2BGR)
    #     case ShownImage.TERMINATIONS:
    #         disp = cv.cvtColor(terminations, cv.COLOR_GRAY2BGR)
    #     case ShownImage.FILTERED_TERMINATIONS:
    #         disp = cv.cvtColor(terminations, cv.COLOR_GRAY2BGR)
    #     case ShownImage.BLUR:
    #         disp = cv.cvtColor(averaged_mask, cv.COLOR_GRAY2BGR)
    #     case ShownImage.SKELETON_MINUS_JUNCTIONS:
    #         disp = cv.cvtColor(split_skeleton, cv.COLOR_GRAY2BGR)

    if draw_junctions:
        # draw the node tree
        for root_idx in tree_roots:
            to_draw = [root_idx]
            while to_draw:
                current_idx = to_draw.pop(0)
                if current_idx in tree:
                    children = tree[current_idx]
                    to_draw.extend([child[0] for child in children])
                if current_idx >= 0 and current_idx in tree:
                    current_pos = (
                        junction_coords[current_idx]
                        if current_idx < len(junction_coords)
                        else termination_coords[current_idx - junction_endpoint_count]
                    )
                    for child_idx, line_idx in tree[current_idx]:
                        child_pos = (
                            junction_coords[child_idx]
                            if child_idx < len(junction_coords)
                            else termination_coords[child_idx - junction_endpoint_count]
                        )
                        cv.arrowedLine(
                            disp,
                            current_pos,
                            child_pos,
                            (255, 255, 0),
                            1,
                        )
                        if draw_terminations:
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

    # if draw_junctions:
    #     disp[expanded_junctions > 0] = (0, 255, 0)
    # if draw_terminations:
    #     disp[terminations > 0] = (255, 0, 255)  # endpoints

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
