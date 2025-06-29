# TERMINOLOGY:
# - junction: a point where 3 or more lines meet
# - termination: a point where 1 line ends
# - node: either a junction or a termination
# - line: A connection between two nodes
# - endpoint: A point on either end of a line which needs to be associated with a node

import os
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

start_drag_x = 0
start_drag_y = 0
end_drag_x = 0
end_drag_y = 0
dragging = False
set_avg = False

min_hsv = np.array([255, 255, 255])
max_hsv = np.array([0, 0, 0])


def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y, start_drag_x, start_drag_y, dragging, set_avg, min_hsv, max_hsv
    # print(event)
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = int(x), int(y)
    elif event == cv.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = int(x), int(y)
        start_drag_x, start_drag_y = int(x), int(y)
        dragging = True
    elif event == cv.EVENT_LBUTTONUP:
        mouse_x, mouse_y = int(x), int(y)
        set_avg = True
        dragging = False
    elif event == cv.EVENT_RBUTTONDOWN:
        mouse_x, mouse_y = int(x), int(y)
        min_hsv = np.array([255, 255, 255])
        max_hsv = np.array([0, 0, 0])


cv.namedWindow(window_title, cv.WINDOW_GUI_NORMAL)
cv.resizeWindow(window_title, 640, 480)
cv.setMouseCallback(window_title, mouse_event)

draw_junctions = False
draw_terminations = False


class ShownImage(Enum):
    TESTING_1 = 0
    TESTING_2 = 1
    TESTING_3 = 2
    TESTING_4 = 3
    RGB = 20
    HSV = 21
    YELLOW = 30
    BLUE = 31
    COMBINED_RAW = 32
    COMBINED = 33
    VORONOI = 40
    LAPLACIAN = 41
    PATH_MASK = 42
    SKELETON = 43
    JUNCTIONS = 44
    TERMINATIONS = 45


# cap = cv.VideoCapture(config.values["hardware"]["camera_id"])
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

prev_offsets = np.zeros(7)

import hardware

hw = hardware.EGB320HardwareAPI(
    port=config.values["hardware"]["port"],
    baudrate=config.values["hardware"]["baudrate"],
)
# hw.open()

test_img_idx = 0

test_img_dir = "photos"

contents = os.listdir(test_img_dir)
photos = []
for content in contents:
    content_path = os.path.join(test_img_dir, content)
    if os.path.isfile(content_path):
        photos.append(content_path)


shown_image = ShownImage.RGB
while True:
    key = cv.waitKey(1)
    if key == ord("-"):
        break
    if key == ord("["):
        draw_junctions = not draw_junctions
    if key == ord("]"):
        draw_terminations = not draw_terminations
    if key == ord("1"):
        shown_image = ShownImage.TESTING_1
    if key == ord("2"):
        shown_image = ShownImage.TESTING_2
    # if key == ord("3"):
    #     shown_image = ShownImage.TESTING_3
    # if key == ord("4"):
    #     shown_image = ShownImage.TESTING_4
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
    if key == ord("v"):
        shown_image = ShownImage.SKELETON
    if key == ord("b"):
        shown_image = ShownImage.JUNCTIONS
    if key == ord("n"):
        shown_image = ShownImage.TERMINATIONS
    if key == ord(","):
        test_img_idx = (test_img_idx - 1) % len(photos)
    if key == ord("."):
        test_img_idx = (test_img_idx + 1) % len(photos)

    if key == ord("R"):
        config.reload()

    # img = cv.imread("paths7.png")
    img = cv.imread(photos[test_img_idx])
    # _, img = cap.read()
    rows, cols, channels = img.shape

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mouse_img_x = min(max(mouse_x, 0), cols - 1)
    mouse_img_y = min(max(mouse_y, 0), rows - 1)
    # print(f"HSV: {img_hsv[mouse_img_y, mouse_img_x]}")

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

    col_denoise_kernel_rad = 3
    color_denoise_kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        ((col_denoise_kernel_rad * 2) - 1, (col_denoise_kernel_rad * 2) - 1),
    )
    combined_mask = cv.morphologyEx(
        combined_raw_mask, cv.MORPH_OPEN, color_denoise_kernel
    )
    combined_mask[rows - 1 - config.values["algorithm"]["sidebar_height"] :, 0:10] = 255
    combined_mask[
        rows - 1 - config.values["algorithm"]["sidebar_height"] :,
        cols - 1 - 10 : cols - 1,
    ] = 255

    fillet_kernel_r = 5
    fillet_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        ((fillet_kernel_r * 2) - 1, (fillet_kernel_r * 2) - 1),
    )
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, fillet_kernel)

    voronoi = cv.distanceTransform(cv.bitwise_not(combined_mask), cv.DIST_L2, 5)

    derivative_kernel_size = 21

    horizontal_sobel = cv.Sobel(voronoi, cv.CV_32F, 2, 0, ksize=derivative_kernel_size)
    vertical_sobel = cv.Sobel(voronoi, cv.CV_32F, 0, 2, ksize=derivative_kernel_size)
    # zero out bottom rows of sobel derivative
    # vertical_sobel[rows - 10 - 1 : rows, :] //= 3
    # laplacian = 255 - cv.normalize(
    #     cv.Laplacian(voronoi, cv.CV_32F, ksize=derivative_kernel_size),
    #     None,
    #     0,
    #     255,
    #     cv.NORM_MINMAX,
    #     cv.CV_8UC1,
    # )
    laplacian = 255 - cv.normalize(
        cv.add(horizontal_sobel, vertical_sobel),
        None,
        0,
        255,
        cv.NORM_MINMAX,
        cv.CV_8UC1,
    )

    voronoi = cv.normalize(
        voronoi,
        None,
        0,
        255,
        cv.NORM_MINMAX,
        cv.CV_8UC1,
    )

    # _, voronoi = cv.threshold(voronoi, 50, 255, cv.THRESH_BINARY)

    # path_mask = cv.adaptiveThreshold(
    #     laplacian, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    _, first_pass_paths = cv.threshold(laplacian, 145, 255, cv.THRESH_BINARY)
    # path_mask[0 : config.values["algorithm"]["blackbar_height"], :] = 0
    path_denoise_kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (3, 3),
    )
    # path_mask = cv.morphologyEx(path_mask, cv.MORPH_OPEN, path_denoise_kernel)

    test_img = cv.multiply(voronoi, first_pass_paths, dtype=cv.CV_16S)
    test_img = cv.normalize(
        test_img,
        None,
        0,
        255,
        cv.NORM_MINMAX,
        cv.CV_8UC1,
    )
    test_thresh = np.max(test_img[rows - 10 - 1 : rows, :]) // 2
    _, test_img_2 = cv.threshold(test_img, test_thresh, 255, cv.THRESH_BINARY)
    second_pass_paths = test_img_2
    # path_mask = test_img

    path_data = pathfinder.find_paths(second_pass_paths)
    tree = path_data["tree"]
    tree_roots = path_data["roots"]
    junction_coords = path_data["junctions"]
    junction_endpoint_count = path_data["junction_endpoint_count"]
    termination_coords = path_data["terminations"]
    line_lengths = path_data["line_lengths"]
    pathfinding_time = path_data["proc_time"]
    # print(f"Pathfinding took {pathfinding_time:.3f} seconds")

    def get_tree_len(node):
        if node >= junction_endpoint_count:
            return 0
        accumulator = 0
        for next_node, line in tree[node]:
            accumulator += line_lengths[line]
            accumulator += get_tree_len(next_node)
        return accumulator

    tree_lens = [int(get_tree_len(x)) for x in tree_roots]
    chosen_root = tree_lens.index(max(tree_lens)) if tree_lens else -1
    # if chosen_root >= 0:
    #     tree_roots = [chosen_root]

    def get_best_node(node_idx, best_idx, current_coords, prev_heuristic=0):
        is_junction = node_idx < junction_endpoint_count
        x, y = (
            junction_coords[node_idx]
            if is_junction
            else termination_coords[node_idx - junction_endpoint_count]
        )
        best_coords = current_coords
        if y < current_coords[1]:
            best_coords = (x, y)
            best_idx = node_idx
        if is_junction:
            for next_node, line in tree[node_idx]:
                (best_idx, best_coords) = get_best_node(
                    next_node, best_idx, best_coords
                )

        # print(best_coords)
        return (best_idx, best_coords)

    best_idx = 0
    best_coords = (1000, 1000)
    if chosen_root >= 0:
        best_idx, best_coords = get_best_node(chosen_root, -1, (1000, 1000))
        # print(best_coords)
        if best_idx >= 0:
            # h_error = best_coords[0] - cols / 2
            h_error = junction_coords[chosen_root][0] - cols / 2
            h_error /= cols / 2
            # print(h_error)
            prev_offsets = np.roll(prev_offsets, 1)
            prev_offsets[0] = h_error
            final_err = np.mean(prev_offsets)
            # print(final_err)
            hw.update(20, -50 * final_err)

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
        case ShownImage.TESTING_1:
            disp = cv.cvtColor(test_img, cv.COLOR_GRAY2BGR)
        case ShownImage.TESTING_2:
            disp = cv.cvtColor(test_img_2, cv.COLOR_GRAY2BGR)
        case ShownImage.TESTING_3:
            disp = cv.cvtColor(test_img_3, cv.COLOR_GRAY2BGR)
        case ShownImage.TESTING_4:
            disp = cv.cvtColor(test_img_4, cv.COLOR_GRAY2BGR)
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
            disp = cv.cvtColor(first_pass_paths, cv.COLOR_GRAY2BGR)
        case ShownImage.SKELETON:
            disp = cv.cvtColor(path_data["skeleton"], cv.COLOR_GRAY2BGR)
        case ShownImage.JUNCTIONS:
            disp = cv.cvtColor(path_data["junction_mask"], cv.COLOR_GRAY2BGR)
        case ShownImage.TERMINATIONS:
            disp = cv.cvtColor(path_data["termination_mask"], cv.COLOR_GRAY2BGR)

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
                            2,
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
        if chosen_root >= 0:
            cv.arrowedLine(
                disp, junction_coords[chosen_root], best_coords, (0, 255, 0), 2
            )

    # if draw_junctions:
    #     disp[expanded_junctions > 0] = (0, 255, 0)
    # if draw_terminations:
    #     disp[terminations > 0] = (255, 0, 255)  # endpoints

    if dragging:
        cv.rectangle(
            disp,
            (start_drag_x, start_drag_y),
            (mouse_x, mouse_y),
            (255, 255, 255),
            1,
        )

    def get_min_max_hsv(start_x, start_y, end_x, end_y):
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        if start_y > end_y:
            start_y, end_y = end_y, start_y
        min_hue = np.min(img_hsv[start_y:end_y, start_x:end_x, 0])
        max_hue = np.max(img_hsv[start_y:end_y, start_x:end_x, 0])
        min_sat = np.min(img_hsv[start_y:end_y, start_x:end_x, 1])
        max_sat = np.max(img_hsv[start_y:end_y, start_x:end_x, 1])
        min_val = np.min(img_hsv[start_y:end_y, start_x:end_x, 2])
        max_val = np.max(img_hsv[start_y:end_y, start_x:end_x, 2])
        return (
            np.array([min_hue, min_sat, min_val]),
            np.array([max_hue, max_sat, max_val]),
        )

    if set_avg:
        set_avg = False
        if start_drag_x == mouse_x or start_drag_y == mouse_y:
            pass
        else:
            min_max = get_min_max_hsv(start_drag_x, start_drag_y, mouse_x, mouse_y)
            print("y", min_max)
            min_hsv = np.minimum(min_hsv, min_max[0])
            max_hsv = np.maximum(max_hsv, min_max[1])
            print("Colour")
            print(f"{((min_hsv + max_hsv) // 2).tolist()}")
            print("Range")
            print(f"{(max_hsv - min_hsv).tolist()}")

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
