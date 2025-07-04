# TERMINOLOGY:
# - junction: a point where 3 or more lines meet
# - termination: a point where 1 line ends
# - node: either a junction or a termination
# - line: A connection between two nodes
# - endpoint: A point on either end of a line which needs to be associated with a node
import render_helpers
import queue
import time
import os
import config
import cv2 as cv
import math
import numpy as np
from enum import Enum
import pathfinder
import asyncio
from datetime import datetime
import threading

# import hardware

print("\r\n\r\nDRC Main Launch File")

print(config.values)

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


q = queue.Queue()


def write_img():
    while True:
        img = q.get()
        # print("w")
        if img is False:
            break
        if not config.values["algorithm"]["use_photos"]:
            cv.imwrite(
                f"archive/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg", img
            )


image_writer = threading.Thread(target=write_img)
image_writer.start()

do_display = config.values["algorithm"]["display"]
if do_display:
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
    MAGENTA = 34
    VORONOI = 40
    LAPLACIAN = 41
    PATH_MASK = 42
    SKELETON = 43
    JUNCTIONS = 44
    TERMINATIONS = 45


cap = None

if not config.values["algorithm"]["use_photos"]:
    cap = cv.VideoCapture(config.values["hardware"]["camera_id"])  #
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

import hardware

hw = hardware.MecanumHardwareAPI(
    port=config.values["hardware"]["port"],
    baudrate=config.values["hardware"]["baudrate"],
)
hw.open()

test_img_idx = 0

test_img_dir = "photos"

contents = os.listdir(test_img_dir)
photos = []
for content in contents:
    content_path = os.path.join(test_img_dir, content)
    if os.path.isfile(content_path):
        photos.append(content_path)

prev_turning_offsets = np.zeros(config.values["algorithm"]["navigation"]["avg_size"])
prev_strafing_offsets = np.zeros(config.values["algorithm"]["navigation"]["avg_size"])
target_coords = None

write_counter = 0
shown_image = ShownImage.RGB
prev_path_tick = cv.getTickCount()
prev_strafe = 0
prev_turn = 0
while True:
    start_ticks = cv.getTickCount()
    key = None
    if do_display:
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
        if key == ord("3"):
            shown_image = ShownImage.TESTING_3
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
        if key == ord("j"):
            shown_image = ShownImage.MAGENTA
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
        if key == ord("Q"):
            target_coords = None

    img = None
    if not config.values["algorithm"]["use_photos"]:
        _, img = cap.read()
    else:
        img = cv.imread(photos[test_img_idx])
    img = img[:, : img.shape[1] - 3, :]
    rows, cols, channels = img.shape
    if not target_coords:
        target_coords = (cols // 2, rows // 2)

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
    magenta_threshold = np.array(
        config.values["algorithm"]["thresholds"]["colors"]["magenta"]
    )
    blue_range = np.array(config.values["algorithm"]["thresholds"]["ranges"]["blue"])
    yellow_range = np.array(
        config.values["algorithm"]["thresholds"]["ranges"]["yellow"]
    )
    magenta_range = np.array(
        config.values["algorithm"]["thresholds"]["ranges"]["magenta"]
    )

    blue_low = blue_threshold - (blue_range / 2)
    blue_high = blue_threshold + (blue_range / 2)
    yellow_low = yellow_threshold - (yellow_range / 2)
    yellow_high = yellow_threshold + (yellow_range / 2)
    magenta_low = magenta_threshold - (magenta_range / 2)
    magenta_high = magenta_threshold + (magenta_range / 2)

    if "blue" in config.values["algorithm"]["thresholds"]["min"]:
        blue_min = np.array(config.values["algorithm"]["thresholds"]["min"]["blue"])
        blue_low = np.maximum(blue_low, blue_min)
    if "blue" in config.values["algorithm"]["thresholds"]["max"]:
        blue_max = np.array(config.values["algorithm"]["thresholds"]["max"]["blue"])
        blue_high = np.minimum(blue_high, blue_max)
    if "yellow" in config.values["algorithm"]["thresholds"]["min"]:
        yellow_min = np.array(config.values["algorithm"]["thresholds"]["min"]["yellow"])
        yellow_low = np.maximum(yellow_low, yellow_min)
    if "yellow" in config.values["algorithm"]["thresholds"]["max"]:
        yellow_max = np.array(config.values["algorithm"]["thresholds"]["max"]["yellow"])
        yellow_high = np.minimum(yellow_high, yellow_max)

    blue_mask = cv.inRange(img_hsv, blue_low, blue_high)
    blue_count = cv.countNonZero(blue_mask)
    yellow_mask = cv.inRange(img_hsv, yellow_low, yellow_high)
    yellow_count = cv.countNonZero(yellow_mask)
    magenta_mask = cv.inRange(img_hsv, magenta_low, magenta_high)
    combined_raw_mask = cv.bitwise_or(
        magenta_mask, cv.bitwise_or(blue_mask, yellow_mask)
    )

    has_blue = blue_count > (
        rows * cols * config.values["algorithm"]["color_loss_threshold"]
    )
    has_yellow = yellow_count > (
        rows * cols * config.values["algorithm"]["color_loss_threshold"]
    )

    open_rad = 1
    color_denoise_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        ((open_rad * 2) - 1, (open_rad * 2) - 1),
    )
    combined_mask = cv.morphologyEx(
        combined_raw_mask, cv.MORPH_OPEN, color_denoise_kernel
    )

    close_rad = 1
    fillet_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        ((close_rad * 2) - 1, (close_rad * 2) - 1),
    )
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, fillet_kernel)
    combined_mask[rows - 1 - config.values["algorithm"]["sidebar_height"] :, 0:10] = 255
    combined_mask[
        rows - 1 - config.values["algorithm"]["sidebar_height"] :,
        cols - 1 - 10 : cols - 1,
    ] = 255

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
    laplacian_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (3, 3),
    )
    laplacian = cv.erode(laplacian, laplacian_kernel, iterations=1)

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
    _, first_pass_paths = cv.threshold(
        laplacian,
        config.values["algorithm"]["pathfinding"]["thresholds"]["laplacian"],
        255,
        cv.THRESH_BINARY,
    )
    first_pass_paths[0 : config.values["algorithm"]["blackbar_height"], :] = 0
    path_denoise_kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (3, 3),
    )

    paths_with_distance = cv.bitwise_and(voronoi, first_pass_paths)
    paths_with_distance = cv.normalize(
        paths_with_distance,
        None,
        0,
        255,
        cv.NORM_MINMAX,
        cv.CV_8UC1,
    )
    path_second_pass_thresh = np.max(paths_with_distance[rows - 10 - 1 : rows, :]) // 2
    _, second_pass_paths = cv.threshold(
        paths_with_distance, path_second_pass_thresh, 255, cv.THRESH_BINARY
    )

    # test_img_3 = voronoi.copy()
    # test_img_3[first_pass_paths == 0] = 0
    # test_img_3 = cv.normalize(
    #     test_img_3,
    #     None,
    #     0,
    #     255,
    #     cv.NORM_MINMAX,
    #     cv.CV_8UC1,
    # )

    path_data = pathfinder.find_paths(second_pass_paths)
    tree = path_data["tree"]
    inverse_tree = path_data["inverse_tree"]
    tree_roots = path_data["roots"]
    tree_endpoint_roots = path_data["endpoint_roots"]
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

    def get_tree_endpoint(node, prev_distance=0):
        best_node = node
        best_distance = prev_distance
        for next_node, line in tree[node]:
            (best_next_node, best_next_distance) = get_tree_endpoint(
                next_node, prev_distance + line_lengths[line]
            )
            best_next_coords = idx_to_coords(best_next_node)
            edge_distance = config.values["algorithm"]["denoising"][
                "image_edge_distance"
            ]
            next_on_edge = (
                # best_next_coords[0] <= edge_distance
                # or best_next_coords[0] >= cols - 1 - edge_distance
                # or
                best_next_coords[1] <= edge_distance
                or best_next_coords[1] >= rows - 1 - edge_distance
            )
            if best_next_distance > best_distance and not next_on_edge:
                best_distance = best_next_distance
                best_node = best_next_node
        return (best_node, best_distance)

    # tree_lens = [int(get_tree_len(x)) for x in tree_roots]
    # chosen_root = tree_lens.index(max(tree_lens)) if tree_lens else -1
    # if chosen_root >= 0:
    #     tree_roots = [chosen_root]

    def heuristic(coords):
        dx = coords[0] - target_coords[0]
        dy = coords[1] - target_coords[1]
        return (dx * dx) + (dy * dy)

    def idx_to_coords(idx):
        if idx < junction_endpoint_count:
            return junction_coords[idx]
        else:
            return termination_coords[idx - junction_endpoint_count]

    best_heuristic = 9999999
    best_idx = -1
    longest_node = -1
    for node in tree.keys():
        coords = idx_to_coords(node)
        this_heuristic = heuristic(coords)
        if this_heuristic < best_heuristic:
            best_heuristic = this_heuristic
            best_idx = node

    # best_idx = 0
    # best_coords = (1000, 1000)
    if best_idx >= 0:
        # best_idx, _ = get_best_node(chosen_root)
        best_coords = idx_to_coords(best_idx)
        longest_node, _ = get_tree_endpoint(best_idx)
        default_x = cols * config.values["algorithm"]["navigation"]["default_target_x"]
        default_y = rows * config.values["algorithm"]["navigation"]["default_target_y"]
        longest_x, longest_y = idx_to_coords(longest_node)

        color_loss_offset = (cols / 2) * config.values["algorithm"]["navigation"][
            "color_loss_offset"
        ]
        history_weighting = config.values["algorithm"]["navigation"][
            "history_weighting"
        ]

        color_offset = 0
        if not has_blue:
            color_offset += color_loss_offset
        if not has_yellow:
            color_offset -= color_loss_offset

        target_x = int(
            default_x + ((longest_x - default_x) * history_weighting) + color_offset
        )
        target_y = int(default_y + ((longest_y - default_y) * history_weighting))

        if target_x < 0:
            target_x = 0
        if target_x >= cols:
            target_x = cols - 1
        if target_y < 0:
            target_y = 0
        if target_y >= rows:
            target_y = rows - 1

        target_coords = (target_x, target_y)

        turning_error = 0
        # h_error = best_coords[0] - cols / 2
        # h_error = junction_coords[chosen_root][0] - cols / 2
        # h_error /= cols / 2
        # print(h_error)

        strafing_error = idx_to_coords(tree_endpoint_roots[best_idx])[0] - cols / 2
        turning_error = longest_x - cols / 2

        strafing_error /= cols / 2
        turning_error /= cols / 2

        prev_turning_offsets = np.roll(prev_turning_offsets, 1)
        prev_turning_offsets[0] = turning_error
        prev_strafing_offsets = np.roll(prev_strafing_offsets, 1)
        prev_strafing_offsets[0] = strafing_error

        avg_turning_err = np.mean(prev_turning_offsets)
        avg_strafing_err = np.mean(prev_strafing_offsets)

        d_turning = avg_turning_err - prev_turn
        d_strafing = avg_strafing_err - prev_strafe
        prev_turn = avg_turning_err
        prev_strafe = avg_strafing_err

        now = cv.getTickCount()
        dt = (now - prev_path_tick) / cv.getTickFrequency()
        prev_path_tick = now

        hw_fwd = config.values["hardware"]["control"]["speeds"]["max"]
        hw_turn = (
            avg_turning_err * config.values["hardware"]["control"]["steering"]["max"]
        )
        hw_strafe = (
            avg_strafing_err * config.values["hardware"]["control"]["strafing"]["max"]
        )

        hw_turn -= d_turning * config.values["hardware"]["pid"]["steering"]["Kd"]
        hw_strafe -= d_strafing * config.values["hardware"]["pid"]["strafing"]["Kd"]

        if abs(hw_fwd) > config.values["hardware"]["limits"]["max_speed"]:
            hw_fwd = config.values["hardware"]["limits"]["max_speed"] * (
                1 if hw_fwd > 0 else -1
            )
        if abs(hw_turn) > config.values["hardware"]["limits"]["max_turn"]:
            hw_turn = config.values["hardware"]["limits"]["max_turn"] * (
                1 if hw_turn > 0 else -1
            )
        if abs(hw_strafe) > config.values["hardware"]["limits"]["max_strafe"]:
            hw_strafe = config.values["hardware"]["limits"]["max_strafe"] * (
                1 if hw_strafe > 0 else -1
            )

        movement_sum = abs(hw_fwd) + abs(hw_turn) + abs(hw_strafe)
        if movement_sum > config.values["hardware"]["limits"]["max_sum"]:
            scale = config.values["hardware"]["limits"]["max_sum"] / movement_sum
            hw_fwd *= scale
            hw_turn *= scale
            hw_strafe *= scale

        if not config.values["algorithm"]["use_photos"]:
            print(f"fwd: {hw_fwd:.2f}, turn: {hw_turn:.2f}, strafe: {hw_strafe:.2f}")

        hw.update(hw_fwd, hw_turn, hw_strafe)

    if do_display:
        disp = img.copy()
        match shown_image:
            case ShownImage.TESTING_1:
                disp = cv.cvtColor(paths_with_distance, cv.COLOR_GRAY2BGR)
            case ShownImage.TESTING_2:
                disp = cv.cvtColor(second_pass_paths, cv.COLOR_GRAY2BGR)
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
            case ShownImage.MAGENTA:
                disp = cv.cvtColor(magenta_mask, cv.COLOR_GRAY2BGR)
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
                        current_pos = idx_to_coords(current_idx)
                        for child_idx, line_idx in tree[current_idx]:
                            child_pos = idx_to_coords(child_idx)
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
            if best_idx >= 0:
                cv.arrowedLine(
                    disp,
                    junction_coords[tree_endpoint_roots[best_idx]],
                    best_coords,
                    (0, 255, 0),
                    2,
                )
                # if best_idx in inverse_tree:
                #     cv.arrowedLine(
                #         disp,
                #         idx_to_coords(inverse_tree[best_idx]),
                #         idx_to_coords(best_idx),
                #         (0, 0, 255),
                #         2,
                #     )
            if longest_node >= 0:
                cv.arrowedLine(
                    disp,
                    idx_to_coords(tree_endpoint_roots[best_idx]),
                    idx_to_coords(longest_node),
                    (0, 0, 255),
                    2,
                )

        if not has_blue:
            render_helpers.render_text(
                disp,
                "No blue",
                (10, 30),
                (255, 255, 255),
            )
        if not has_yellow:
            render_helpers.render_text(
                disp,
                "No yellow",
                (10, 60),
                (255, 255, 255),
            )
        cv.drawMarker(
            disp,
            target_coords,
            (255, 255, 255),
            markerType=cv.MARKER_CROSS,
            markerSize=10,
            thickness=2,
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
    end_ticks = cv.getTickCount()
    elapsed_time = (end_ticks - start_ticks) / cv.getTickFrequency()

    if not config.values["algorithm"]["use_photos"]:
        print(f"dt: {elapsed_time:.3f}s")

    write_counter += 1
    if write_counter >= 5:
        write_counter = 0
        q.put(img.copy())
q.put(False)
image_writer.join()
cv.destroyAllWindows()
