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

mouse_x = 0
mouse_y = 0


def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = int(x), int(y)


cv.namedWindow(window_title, cv.WINDOW_GUI_NORMAL)
cv.setMouseCallback(window_title, mouse_event)

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
    RGB = 20
    HSV = 21
    YELLOW = 30
    BLUE = 31
    COMBINED_RAW = 32
    COMBINED = 33
    VORONOI = 40
    LAPLACIAN = 41
    PATH_MASK = 42


cap = cv.VideoCapture(config.values["hardware"]["camera_id"])
if not cap.isOpened():
    print("Cannot open camera")
    exit()

shown_image = ShownImage.INPUT
while True:
    key = cv.waitKey(1)
    if key == ord("-"):
        break
    if key == ord("["):
        draw_junctions = not draw_junctions
    if key == ord("]"):
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

    if key == ord("a"):
        shown_image = ShownImage.RGB
    if key == ord("s"):
        shown_image = ShownImage.HSV
    if key == ord("d"):
        shown_image = ShownImage.YELLOW
    if key == ord("f"):
        shown_image = ShownImage.BLUE
    if key == ord("g"):
        shown_image = ShownImage.COMBINED_RAW
    if key == ord("h"):
        shown_image = ShownImage.COMBINED
    if key == ord("z"):
        shown_image = ShownImage.VORONOI
    if key == ord("x"):
        shown_image = ShownImage.LAPLACIAN
    if key == ord("c"):
        shown_image = ShownImage.PATH_MASK

    # img = cv.imread("paths7.png")
    _, img = cap.read()
    rows, cols, channels = img.shape

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mouse_img_x = min(max(mouse_x, 0), cols - 1)
    mouse_img_y = min(max(mouse_y, 0), rows - 1)
    print(f"HSV: {img_hsv[mouse_img_y, mouse_img_x]}")

    blue_threshold = np.array(
        config.values["algorithm"]["thresholds"]["colors"]["blue"]
    )
    yellow_threshold = np.array(
        config.values["algorithm"]["thresholds"]["colors"]["yellow"]
    )
    blue_range = np.array(config.values["algorithm"]["thresholds"]["ranges"]["blue"])
    yellow_range = np.array(
        config.values["algorithm"]["thresholds"]["ranges"]["yellow"]
    )

    blue_low = blue_threshold - (blue_range / 2)
    blue_high = blue_threshold + (blue_range / 2)
    yellow_low = yellow_threshold - (yellow_range / 2)
    yellow_high = yellow_threshold + (yellow_range / 2)

    blue_mask = cv.inRange(img_hsv, blue_low, blue_high)
    yellow_mask = cv.inRange(img_hsv, yellow_low, yellow_high)
    combined_raw_mask = cv.bitwise_or(blue_mask, yellow_mask)

    col_denoise_kernel_rad = 2
    color_denoise_kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        ((col_denoise_kernel_rad * 2) - 1, (col_denoise_kernel_rad * 2) - 1),
    )
    combined_mask = cv.morphologyEx(
        combined_raw_mask, cv.MORPH_OPEN, color_denoise_kernel
    )
    combined_mask[rows - 1 - config.values["algorithm"]["sidebar_height"] :, 0] = 255
    combined_mask[
        rows - 1 - config.values["algorithm"]["sidebar_height"] :, cols - 1
    ] = 255

    voronoi = cv.distanceTransform(cv.bitwise_not(combined_mask), cv.DIST_L2, 5)

    derivative_kernel_size = 21
    laplacian = cv.normalize(
        cv.Laplacian(voronoi, cv.CV_32F, ksize=derivative_kernel_size),
        None,
        0,
        255,
        cv.NORM_MINMAX,
        cv.CV_8UC1,
    )

    # path_mask = cv.adaptiveThreshold(
    #     laplacian, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    _, path_mask = cv.threshold(laplacian, 100, 255, cv.THRESH_BINARY_INV)

    # input_mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # path_data = pathfinder.find_paths(input_mask)
    # tree = path_data["tree"]
    # tree_roots = path_data["roots"]
    # junction_coords = path_data["junctions"]
    # junction_endpoint_count = path_data["junction_endpoint_count"]
    # termination_coords = path_data["terminations"]
    # line_lengths = path_data["line_lengths"]
    # pathfinding_time = path_data["proc_time"]
    # print(f"Pathfinding took {pathfinding_time:.3f} seconds")

    disp = img.copy()
    match shown_image:
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
        case ShownImage.RGB:
            disp = img.copy()
        case ShownImage.HSV:
            disp = img_hsv.copy()
        case ShownImage.BLUE:
            disp = cv.cvtColor(blue_mask, cv.COLOR_GRAY2BGR)
        case ShownImage.YELLOW:
            disp = cv.cvtColor(yellow_mask, cv.COLOR_GRAY2BGR)
        case ShownImage.COMBINED_RAW:
            disp = cv.cvtColor(combined_raw_mask, cv.COLOR_GRAY2BGR)
        case ShownImage.COMBINED:
            disp = cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)
        case ShownImage.VORONOI:
            normalised = cv.normalize(
                voronoi,
                None,
                0,
                255,
                cv.NORM_MINMAX,
                cv.CV_8UC1,
            )
            disp = cv.cvtColor(normalised, cv.COLOR_GRAY2BGR)
        case ShownImage.LAPLACIAN:
            normalised = cv.normalize(
                laplacian,
                None,
                0,
                255,
                cv.NORM_MINMAX,
                cv.CV_8UC1,
            )
            disp = cv.cvtColor(normalised, cv.COLOR_GRAY2BGR)
        case ShownImage.PATH_MASK:
            disp = cv.cvtColor(path_mask, cv.COLOR_GRAY2BGR)

    # if draw_junctions:
    #     # draw the node tree
    #     for root_idx in tree_roots:
    #         to_draw = [root_idx]
    #         while to_draw:
    #             current_idx = to_draw.pop(0)
    #             if current_idx in tree:
    #                 children = tree[current_idx]
    #                 to_draw.extend([child[0] for child in children])
    #             if current_idx >= 0 and current_idx in tree:
    #                 current_pos = (
    #                     junction_coords[current_idx]
    #                     if current_idx < len(junction_coords)
    #                     else termination_coords[current_idx - junction_endpoint_count]
    #                 )
    #                 for child_idx, line_idx in tree[current_idx]:
    #                     child_pos = (
    #                         junction_coords[child_idx]
    #                         if child_idx < len(junction_coords)
    #                         else termination_coords[child_idx - junction_endpoint_count]
    #                     )
    #                     cv.arrowedLine(
    #                         disp,
    #                         current_pos,
    #                         child_pos,
    #                         (255, 255, 0),
    #                         1,
    #                     )
    #                     if draw_terminations:
    #                         cv.putText(
    #                             disp,
    #                             str(line_lengths[line_idx]),
    #                             (
    #                                 (current_pos[0] + child_pos[0]) // 2,
    #                                 (current_pos[1] + child_pos[1]) // 2,
    #                             ),
    #                             cv.FONT_HERSHEY_SIMPLEX,
    #                             0.5,
    #                             (255, 0, 255),
    #                         )

    # if draw_junctions:
    #     disp[expanded_junctions > 0] = (0, 255, 0)
    # if draw_terminations:
    #     disp[terminations > 0] = (255, 0, 255)  # endpoints

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
