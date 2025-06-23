import cv2 as cv
import numpy as np
import time
import math
import os
import platform
from enum import IntEnum
from multiprocessing import Process, Value
from simple_pid import PID
from crc import Calculator, Crc8
import serial
import struct

calculator = Calculator(Crc8.CCITT)
is_on_pc = platform.system() == "Windows"
remote_display = True or is_on_pc

process_scale = 1 if is_on_pc else 1 / 2
display_scale = 1 / process_scale
display_scale = display_scale * (1 if remote_display else 0.5)
ser_send = True
# ser_send = False
os.environ["DISPLAY"] = "192.168.81.74:0" if remote_display else ":0"

window_title = "DRC Pathfinder"
# ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)


class DisplayMode(IntEnum):
    RGB = 0
    HUE = 1
    SAT = 2
    VAL = 3
    HSV = 4
    MASK_BLUE = -1
    MASK_YELLOW = -2
    MASK_MAGENTA = -3
    MASK_RED = -4
    MASK_COMBINED = -5
    MASK_PATH = -6
    VORONOI_DIST = -10
    H_SOBOL = -11
    V_SOBOL = -12
    COMBINED_DERIV = -13
    RAW_PATHS = -20
    DENOISED_PATHS = -21
    FINAL_PATHS = -22
    CHOSEN_PATH = -23
    CONTOURS = -24


def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = int(x), int(y)


def process_key(key):
    global current_display_mode
    global robot_stop
    global program_start

    if key == ord("g"):
        current_display_mode = DisplayMode.RGB
    if key == ord("h"):
        current_display_mode = DisplayMode.HUE
    if key == ord("j"):
        current_display_mode = DisplayMode.SAT
    if key == ord("k"):
        current_display_mode = DisplayMode.VAL
    if key == ord("l"):
        current_display_mode = DisplayMode.HSV
    if key == ord("1"):
        current_display_mode = DisplayMode.MASK_BLUE
    if key == ord("2"):
        current_display_mode = DisplayMode.MASK_YELLOW
    if key == ord("3"):
        current_display_mode = DisplayMode.MASK_MAGENTA
    if key == ord("4"):
        current_display_mode = DisplayMode.MASK_RED
    if key == ord("5"):
        current_display_mode = DisplayMode.MASK_COMBINED
    if key == ord("6"):
        current_display_mode = DisplayMode.MASK_PATH
    if key == ord("a"):
        current_display_mode = DisplayMode.VORONOI_DIST
    if key == ord("s"):
        current_display_mode = DisplayMode.H_SOBOL
    if key == ord("d"):
        current_display_mode = DisplayMode.V_SOBOL
    if key == ord("f"):
        current_display_mode = DisplayMode.COMBINED_DERIV
    if key == ord("z"):
        current_display_mode = DisplayMode.RAW_PATHS
    if key == ord("x"):
        current_display_mode = DisplayMode.DENOISED_PATHS
    if key == ord("c"):
        current_display_mode = DisplayMode.FINAL_PATHS
    if key == ord("v"):
        current_display_mode = DisplayMode.CHOSEN_PATH
    if key == ord("b"):
        current_display_mode = DisplayMode.CONTOURS
    if key == ord(" "):
        robot_stop = True
    if key == ord("`"):
        program_start = time.monotonic()
        robot_stop = False


def render_text(img, text, org, col=(0, 0, 0), border=(255, 255, 255), scale=1):
    scale = scale * process_scale
    (x, y) = org
    org = (int(x * process_scale), int(y * process_scale))
    cv.putText(
        img,
        text,
        org,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        border,
        math.ceil(3 * scale),
        cv.LINE_AA,
    )
    cv.putText(
        img,
        text,
        org,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        col,
        math.ceil(1 * scale),
        cv.LINE_AA,
    )


def steering_to_motor_vals(steering, kill):
    steering = min(max(steering, -1), 1)
    min_forward = 0.6
    max_forward = min_forward

    max_steering = 0.6

    min_speed_steering = 0.3

    slow_lerp = min(abs(steering) / min_speed_steering, 1)

    forward_out = ((1 - slow_lerp) * max_forward) + (slow_lerp * min_forward)
    steering_out = steering * max_steering

    if kill:
        return (0, 0)

    left = forward_out - steering_out
    right = forward_out + steering_out
    return (left, right)


def serial_io_loop(
    current_error,
    future_error,
    current_avg,
    future_avg,
    current_target,
    current_pid,
    lookahead_proportion,
    path_lost,
    kill,
):
    target_ups = 50
    averaging_time = 0.1
    mot_averaging_time = 0.1
    lookahead_start = 0.3

    Kp = 3.0
    Kd = 0.5
    Ki = 0
    pid = PID(Kp, Ki, Kd, setpoint=0, output_limits=(-1, 1))
    motor_range = 255

    # input averaging
    averaging_count = math.ceil(target_ups * averaging_time)
    mot_averaging_count = math.ceil(target_ups * mot_averaging_time)
    current_avg_buf = np.zeros(averaging_count)
    future_avg_buf = np.zeros(averaging_count)
    left_avg_buf = np.zeros(mot_averaging_count)
    right_avg_buf = np.zeros(mot_averaging_count)

    # average weighting function
    weights = [(((x + 1) / averaging_count) ** 2) for x in range(0, averaging_count)]

    while not kill.value:
        exec_start = time.monotonic()

        current_avg_buf = np.roll(current_avg_buf, -1)
        current_avg_buf[-1] = current_error.value
        current_avg_out = np.average(current_avg_buf, weights=weights)
        current_avg.value = current_avg_out

        future_avg_buf = np.roll(future_avg_buf, -1)
        future_avg_buf[-1] = future_error.value
        future_avg_out = np.average(future_avg_buf, weights=weights)
        future_avg.value = future_avg_out

        # 2 ^ (-a*x^2) curve (bell curve)
        lerp_val = 2 ** (-5 * ((abs(current_avg_out) / lookahead_start) ** 2))
        lookahead_proportion.value = lerp_val

        # current_target_out = ((1 - lerp_val) * current_avg_out) + (
        #     lerp_val * future_avg_out
        # )
        current_target_out = future_avg_out
        current_target.value = current_target_out

        pid_out = pid(current_target_out)
        current_pid.value = pid_out

        exec_end = time.monotonic()
        dt = exec_end - exec_start
        left_f, right_f = steering_to_motor_vals(pid_out, path_lost.value)
        left = int(left_f * motor_range)

        left = int(left_f * motor_range)
        left_avg_buf = np.roll(left_avg_buf, -1)
        left_avg_buf[-1] = left
        left_avg = int(np.average(left_avg_buf))

        right = int(right_f * motor_range)
        right_avg_buf = np.roll(right_avg_buf, -1)
        right_avg_buf[-1] = right
        right_avg = int(np.average(right_avg_buf))

        print(
            f"P:{pid_out:+01.3f}, L{left:+06d}, R{right:+06d}, C{current_avg_out:+01.3f}, F{future_avg_out:+01.3f}"
        )
        raw_data = struct.pack("<Bll", 0xAA, left_avg, right_avg)
        crc = calculator.checksum(raw_data)
        final_data = struct.pack("<BllB", 0xAA, left_avg, right_avg, crc)
        if ser_send:
            ser.write(final_data)

        time.sleep(max((1.0 / target_ups) - dt, 0))


if __name__ == "__main__":
    print("\r\n\r\n")
    color_min_area_proportion = 0.01 * 0.1
    no_col_detect_error = 0.05
    future_path_height = 0.35
    show = True

    # DEBUGGING + DISPLAY
    current_display_mode = DisplayMode.RGB
    robot_stop = remote_display

    mouse_x = 0
    mouse_y = 0

    # THRESHOLDING
    blu_hsv = (146, 234, 161)
    ylw_hsv = (45, 211, 175)
    mgnta_hsv = (230, 130, 70)
    red_hsv = (0, 0, 0)

    blu_hsv_thresh_range = (20, 60, 60)
    ylw_hsv_thresh_range = (20, 30, 60)
    mgnta_hsv_thresh_range = (40, 60, 50)
    red_hsv_thresh_range = (0, 0, 0)

    col_denoise_kernel_rad = 2
    edge_fill_height = 0.25

    # DENOISING/CLEANDING KERNELS
    color_denoise_kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        ((col_denoise_kernel_rad * 2) - 1, (col_denoise_kernel_rad * 2) - 1),
    )

    path_open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # path_dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 11))
    path_dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # PARAMETERS
    derivative_kernel_size = 21
    path_threshold_val = 90
    path_slice_height_px = 10

    initial_blur_size = 31

    # more image proportions

    # filtering/slicing image proportions
    horizontal_cutoff_dist = 0.5
    path_min_area_proportion = 0.01 * 0.1

    # robot config parameters
    path_failsafe_time = 0.3

    setpoint = 0

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    current_error_val = Value("d", 0)
    future_error_val = Value("d", 0)
    current_avg_val = Value("d", 0)
    future_avg_val = Value("d", 0)

    current_target_val = Value("d", 0)
    current_pid_val = Value("d", 0)
    lookahead_proportion_val = Value("d", 0)
    path_lost_val = Value("i", 0)
    left_val = Value("i", 0)
    right_val = Value("i", 0)
    serial_should_stop = Value("i", 0)
    program_start = time.monotonic()
    path_lost_val.value = True

    serial_io_thread = Process(
        target=serial_io_loop,
        args=(
            current_error_val,
            future_error_val,
            current_avg_val,
            future_avg_val,
            current_target_val,
            current_pid_val,
            lookahead_proportion_val,
            path_lost_val,
            serial_should_stop,
        ),
    )
    # serial_io_thread.start()

    picIn = cv.imread("test2.png")  # TODO TESTING ONLY
    # _, picIn = cap.read()
    rows, cols, channels = picIn.shape
    rows = int(rows * process_scale)
    cols = int(cols * process_scale)

    image_centre_x = int(cols / 2)
    image_centre_y = int(rows / 2)
    pathfinding_centre_x = image_centre_x
    horizontal_cutoff_dist_px = int(horizontal_cutoff_dist * cols / 2)
    future_path_offset_px = int(rows * future_path_height)
    path_minimum_area = int(path_min_area_proportion * rows * cols)
    color_min_area = int(color_min_area_proportion * rows * cols)
    no_col_detect_error_px = int(cols * no_col_detect_error)

    bottom_slice_mask = np.zeros((rows, cols), np.uint8)
    bottom_slice_mask[rows - path_slice_height_px : rows, :] = 1
    path_cutoff_height = rows - path_slice_height_px
    bottom_row_clear_px = 5
    edge_fill_height_px = int(rows * edge_fill_height)

    setpoint_px = int(image_centre_x + (setpoint * cols / 2))

    blu_hsv_low = np.array(
        [pair[0] - pair[1] for pair in zip(blu_hsv, blu_hsv_thresh_range)]
    )
    blu_hsv_high = np.array(
        [pair[0] + pair[1] for pair in zip(blu_hsv, blu_hsv_thresh_range)]
    )

    ylw_hsv_low = np.array(
        [pair[0] - pair[1] for pair in zip(ylw_hsv, ylw_hsv_thresh_range)]
    )
    ylw_hsv_high = np.array(
        [pair[0] + pair[1] for pair in zip(ylw_hsv, ylw_hsv_thresh_range)]
    )

    mgnta_hsv_low = np.array(
        [pair[0] - pair[1] for pair in zip(mgnta_hsv, mgnta_hsv_thresh_range)]
    )
    mgnta_hsv_high = np.array(
        [pair[0] + pair[1] for pair in zip(mgnta_hsv, mgnta_hsv_thresh_range)]
    )

    red_hsv_low = np.array(
        [pair[0] - pair[1] for pair in zip(red_hsv, red_hsv_thresh_range)]
    )
    red_hsv_high = np.array(
        [pair[0] + pair[1] for pair in zip(red_hsv, red_hsv_thresh_range)]
    )

    path_max_y_check = int(rows * 0.49)
    path_mask_widen_end_y = int(rows * 0.5)
    path_mask_widen_start_y = int(rows * 0.95)
    path_mask = np.zeros((rows, cols), np.uint8)
    path_mask_contour = np.array(
        [
            # [0, 0],
            # [cols - 1, 0],
            # [cols - 1, path_mask_y2],
            # [0, path_mask_y1],
            [pathfinding_centre_x - horizontal_cutoff_dist_px, rows - 1],
            [pathfinding_centre_x - horizontal_cutoff_dist_px, path_mask_widen_start_y],
            [0, path_mask_widen_end_y],
            [0, path_max_y_check],
            [cols - 1, path_max_y_check],
            [cols - 1, path_mask_widen_end_y],
            [pathfinding_centre_x + horizontal_cutoff_dist_px, path_mask_widen_start_y],
            [pathfinding_centre_x + horizontal_cutoff_dist_px, rows - 1],
        ],
        dtype=np.int32,
    )
    path_mask_contour = path_mask_contour.reshape((-1, 1, 2))
    # cv.polylines(path_mask, [path_mask_contour], True, 255)
    cv.drawContours(path_mask, [path_mask_contour], 0, 255, -1)

    # FAILSAFE
    last_time_path_seen = time.monotonic()

    # OPENCV WINDOW
    cv.namedWindow(window_title, cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow(window_title, int(cols * display_scale), int(rows * display_scale))
    if not remote_display:
        cv.moveWindow(window_title, 0, -20)
    cv.setMouseCallback(window_title, mouse_event)

    while True:
        key = cv.waitKey(1)
        process_key(key)
        if key == ord("-"):
            break
        if time.monotonic() - program_start > 10 and not remote_display:
            break

        start_time = time.monotonic()
        # IMAGE INPUT
        input_frame = picIn.copy()
        # ret, input_frame = cap.read()
        # if ret == False:
        #     break
        capture_time = time.monotonic()
        # no need to blur if the camera's defocused!
        # input_frame = cv.blur(input_frame, (initial_blur_size, initial_blur_size))
        # input_frame = cv.GaussianBlur(
        #     input_frame, (initial_blur_size, initial_blur_size), 0
        # )
        # scale image down to reduce processing time at the cost of error resolution
        input_frame = cv.resize(input_frame, (cols, rows))
        # input_frame = cv.flip(input_frame, -1)

        hsvImg = cv.cvtColor(input_frame, cv.COLOR_BGR2HSV_FULL)

        # calculate thresholded masks for various colors
        blu_mask = cv.inRange(hsvImg, blu_hsv_low, blu_hsv_high)
        blu_detected = cv.countNonZero(blu_mask) > color_min_area
        blu_mask[
            rows - edge_fill_height_px : rows, cols - col_denoise_kernel_rad - 1 : cols
        ] = 255
        ylw_mask = cv.inRange(hsvImg, ylw_hsv_low, ylw_hsv_high)
        ylw_detected = cv.countNonZero(ylw_mask) > color_min_area
        ylw_mask[rows - edge_fill_height_px : rows, 0 : col_denoise_kernel_rad + 1] = (
            255
        )
        mgnta_mask = cv.inRange(hsvImg, mgnta_hsv_low, mgnta_hsv_high)
        # combine the masks
        track_boundaries_mask = cv.bitwise_xor(ylw_mask, blu_mask)
        avoid_mask = cv.bitwise_xor(track_boundaries_mask, mgnta_mask)
        # and denoise
        avoid_mask = cv.morphologyEx(avoid_mask, cv.MORPH_OPEN, color_denoise_kernel)

        # distanceTransform gives distance from nearest *zero pixel*, not from nearest *white* pixel, so it needs to be inverted
        # ksize 3 gives really bad results for some reason even though it is slightly faster
        distance_plot = cv.distanceTransform(cv.bitwise_not(avoid_mask), cv.DIST_L2, 5)

        # take second-order horizontal Sobel derivative of the image
        # This converts the "peaks" in the Voronoi diagram into significantly negative regions
        # or, after normalisation, local (and global!) minima
        raw_horz_derivative = cv.Sobel(
            distance_plot, cv.CV_32F, 2, 0, ksize=derivative_kernel_size
        )
        horz_d1 = cv.Sobel(
            distance_plot, cv.CV_32F, 1, 0, ksize=derivative_kernel_size
        )
        raw_vert_derivative = cv.Sobel(
            distance_plot, cv.CV_32F, 0, 2, ksize=derivative_kernel_size
        )
        # laplacian is just horiz + vertical 2nd order sobel added together
        # allows it to handle sharp corners or U-turns better
        raw_derivative = (raw_horz_derivative * 1.1) + (raw_vert_derivative * 0.5)
        # raw_derivative = cv.Laplacian(
        #     distance_plot, cv.CV_32F, ksize=derivative_kernel_size
        # )

        # normalise the insane values that the derivative produces to u8 range
        normalised_derivative = cv.normalize(
            raw_derivative, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1
        )
        # only the minimum is needed, but maybe in the future use the coords output to auto-choose the best path?
        # would mean no more weighting though...
        # bottom slice just used to prevent any weirdness in the top part of the image from throwing the results
        (
            minimum_derivative,
            maximum_derivative,
            minimum_derivative_coords,
            maximum_derivative_coords,
        ) = cv.minMaxLoc(normalised_derivative, bottom_slice_mask)

        # threshold based on the minimum found gets us the "ridgelines" in the distance plot
        _, raw_paths_binary = cv.threshold(
            normalised_derivative, path_threshold_val, 255, cv.THRESH_BINARY_INV
        )
        # opening (erode/dilate) removes the "strings" produced by the diagonal lines
        # denoised_paths_mask = cv.morphologyEx(
        #     raw_paths_mask, cv.MORPH_OPEN, path_open_kernel
        # )
        # finally a mostly-vertical dilation re-joins paths that sometimes split after the open operation
        # final_paths_mask = cv.dilate(denoised_paths_mask, path_dilate_kernel)
        final_paths_binary = cv.dilate(raw_paths_binary, path_dilate_kernel)
        final_paths_binary = cv.bitwise_and(final_paths_binary, path_mask)
        final_paths_binary[rows - bottom_row_clear_px : rows, :] = final_paths_binary[
            rows - bottom_row_clear_px - 1, :
        ]

        # find all the contours - since they should all be separate lines and only one is chosen,
        # heirarchy can get thrown away
        contours, _ = cv.findContours(
            final_paths_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )

        # centroids = [None] * len(contours)
        bounding_boxes = [None] * len(contours)
        potential_path_contours = []

        chosen_path_idx = -1
        closest_path_dist = 9999999

        # TODO maybe could cut this down to only the bottom/middle slices?
        # might interfere with bounding boxes but could be decent optimisation

        # locate valid potential path contours
        for i in range(len(contours)):
            contour = contours[i]
            # calculate x/y centre of contour using moments
            contour_moment = cv.moments(contour, True)
            # add 1e-5 to avoid division by zero
            centroid_x = int(contour_moment["m10"] / (contour_moment["m00"] + 1e-5))
            # centroid_y = int(contour_moment["m01"] / (contour_moment["m00"] + 1e-5))
            # centroids[i] = (centroid_x, centroid_y)

            # calculate bounding box coordinates
            bounding_box = cv.boundingRect(contour)
            bounding_boxes[i] = bounding_box
            tr_x, tr_y, w, h = bounding_box
            bl_x = tr_x + w
            bl_y = tr_y + h

            path_dist = min(
                abs(pathfinding_centre_x - bl_x),
                abs(pathfinding_centre_x - tr_x),
                abs(pathfinding_centre_x - centroid_x),
            )
            # if the path starts in the bottom rows,
            # AND if any of the contour corners or its centroid are
            # within range of the path choice centreline
            # AND it's not tiny
            if (
                bl_y >= path_cutoff_height
                and path_dist < horizontal_cutoff_dist_px
                and cv.contourArea(contour) > path_minimum_area
            ):
                # then it's an option for a path
                potential_path_contours.append(i)
                # want to choose the path closest to the current path centreline
                # if it's closer, update the new distance and index
                if path_dist < closest_path_dist:
                    closest_path_dist = path_dist
                    chosen_path_idx = i

        # create a mask image with only the selected path in it
        chosen_path_binary = np.zeros_like(final_paths_binary)

        # pathfinding output vars
        current_path_x = pathfinding_centre_x
        future_path_x = current_path_x
        path_lost = False
        short_path_warn = True

        # robot failsafe stop
        should_stop = False

        # failsafe if no path detected
        if chosen_path_idx < 0:
            if (time.monotonic() - last_time_path_seen) > path_failsafe_time:
                should_stop = True
            path_lost = True
        else:
            last_time_path_seen = time.monotonic()
            cv.drawContours(
                chosen_path_binary, contours, chosen_path_idx, 255, cv.FILLED
            )

            # and then slice that and use moments to get the x coordinates of the start and middle of the path
            current_path_slice = chosen_path_binary[
                (rows - path_slice_height_px) : rows, :
            ]
            current_path_moment = cv.moments(current_path_slice, True)
            current_path_x = int(
                current_path_moment["m10"] / (current_path_moment["m00"] + 1e-5)
            )
            future_path_slice_start = rows - future_path_offset_px
            future_path_slice = chosen_path_binary[
                future_path_slice_start : (
                    future_path_slice_start + path_slice_height_px
                ),
                :,
            ]
            future_path_moment = cv.moments(
                future_path_slice,
                True,
            )
            future_path_x = current_path_x
            if future_path_moment["m10"] > 1:
                future_path_x = int(
                    future_path_moment["m10"] / (future_path_moment["m00"] + 1e-5)
                )
                short_path_warn = False

        end_time = time.monotonic()

        # RENDERING
        match current_display_mode:
            case DisplayMode.RGB:
                txt = "RGB"
                display_frame = input_frame
            case DisplayMode.HUE:
                txt = "Hue"
                hsvHueOnly = hsvImg.copy()
                hsvHueOnly[:, :, 1] = 255
                hsvHueOnly[:, :, 2] = 255
                hsvHueOnly = cv.cvtColor(hsvHueOnly, cv.COLOR_HSV2BGR_FULL)
                display_frame = hsvHueOnly
            case DisplayMode.SAT:
                txt = "Sat"
                display_frame = cv.cvtColor(hsvImg[:, :, 1], cv.COLOR_GRAY2BGR)
            case DisplayMode.VAL:
                txt = "Val"
                display_frame = cv.cvtColor(hsvImg[:, :, 2], cv.COLOR_GRAY2BGR)
            case DisplayMode.HSV:
                txt = "HSV"
                display_frame = hsvImg

            case DisplayMode.MASK_BLUE:
                txt = "BLU MASK"
                display_frame = cv.cvtColor(blu_mask, cv.COLOR_GRAY2BGR)
            case DisplayMode.MASK_YELLOW:
                txt = "YLW MASK"
                display_frame = cv.cvtColor(ylw_mask, cv.COLOR_GRAY2BGR)
            case DisplayMode.MASK_MAGENTA:
                txt = "MAG MASK"
                display_frame = cv.cvtColor(mgnta_mask, cv.COLOR_GRAY2BGR)
            case DisplayMode.MASK_RED:
                txt = "RED MSK"
                display_frame = cv.cvtColor(avoid_mask, cv.COLOR_GRAY2BGR)
                # display_frame = cv.cvtColor(red_mask, cv.COLOR_GRAY2BGR) # TODO
            case DisplayMode.MASK_COMBINED:
                txt = "CMB MSK"
                display_frame = cv.cvtColor(avoid_mask, cv.COLOR_GRAY2BGR)
            case DisplayMode.MASK_PATH:
                txt = "PATH MASK"
                display_frame = cv.cvtColor(path_mask, cv.COLOR_GRAY2BGR)

            case DisplayMode.VORONOI_DIST:
                txt = "DIST"
                combinedDistFrame = cv.normalize(
                    distance_plot,
                    None,
                    0,
                    255,
                    cv.NORM_MINMAX,
                    cv.CV_8UC1,
                )
                display_frame = cv.cvtColor(combinedDistFrame, cv.COLOR_GRAY2BGR)
            case DisplayMode.H_SOBOL:
                txt = "HSBL"
                display_frame = cv.cvtColor(
                    cv.normalize(
                        raw_horz_derivative, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1
                    ),
                    cv.COLOR_GRAY2BGR,
                )
            case DisplayMode.V_SOBOL:
                txt = "VSBL"
                display_frame = cv.cvtColor(
                    cv.normalize(
                        horz_d1, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1
                    ),
                    cv.COLOR_GRAY2BGR,
                )
            case DisplayMode.COMBINED_DERIV:
                txt = "DERV"
                display_frame = cv.cvtColor(normalised_derivative, cv.COLOR_GRAY2BGR)

            case DisplayMode.RAW_PATHS:
                txt = "RAW PATHS"
                display_frame = cv.cvtColor(raw_paths_binary, cv.COLOR_GRAY2BGR)
            case DisplayMode.DENOISED_PATHS:
                txt = "DNSD PATHS"
                display_frame = cv.cvtColor(
                    final_paths_binary, cv.COLOR_GRAY2BGR
                )  # TODO
            case DisplayMode.CONTOURS:
                txt = "CONTOURS"
                display_frame = cv.cvtColor(final_paths_binary, cv.COLOR_GRAY2BGR)
                for x, y, w, h in bounding_boxes:
                    cv.rectangle(
                        display_frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                cv.drawContours(display_frame, contours, -1, (0, 255, 0), 1, cv.LINE_AA)
            case DisplayMode.FINAL_PATHS:
                txt = "FNL PATHS"
                display_frame = cv.cvtColor(final_paths_binary, cv.COLOR_GRAY2BGR)
                cv.drawContours(
                    display_frame,
                    contours,
                    chosen_path_idx,
                    (255, 255, 0),
                    -1,
                    cv.LINE_AA,
                )
                for i in potential_path_contours:
                    cv.drawContours(
                        display_frame,
                        contours,
                        i,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
            case DisplayMode.CHOSEN_PATH:
                txt = "CSN PATH"
                display_frame = cv.cvtColor(chosen_path_binary, cv.COLOR_GRAY2BGR)

            case _:
                txt = "Dflt"
                display_frame = input_frame

        current_path_error_px = current_path_x - setpoint_px
        future_path_error_px = future_path_x - setpoint_px
        if not blu_detected:
            current_path_error_px = current_path_error_px + no_col_detect_error_px
            future_path_error_px = future_path_error_px + no_col_detect_error_px
        if not ylw_detected:
            current_path_error_px = current_path_error_px - no_col_detect_error_px
            future_path_error_px = future_path_error_px - no_col_detect_error_px

        path_lost_val.value = path_lost or robot_stop
        current_error_val.value = current_path_error_px / (cols / 2)
        future_error_val.value = future_path_error_px / (cols / 2)

        current_avg = pathfinding_centre_x + int(current_avg_val.value * (cols / 2))
        future_avg = current_avg + int(future_avg_val.value * (cols / 2))
        lookahead_proportion = int(
            lookahead_proportion_val.value * future_path_offset_px
        )
        current_target = pathfinding_centre_x + int(
            current_target_val.value * (cols / 2)
        )

        # draw calculated pathfinding markers on final image
        # show image centreline/path border
        cv.line(
            display_frame,
            (pathfinding_centre_x, 0),
            (pathfinding_centre_x, rows),
            (255, 0, 255),
            2,
        )
        cv.line(
            display_frame,
            (pathfinding_centre_x - horizontal_cutoff_dist_px, rows - 1),
            (pathfinding_centre_x + horizontal_cutoff_dist_px, rows - 1),
            (255, 0, 255),
            4,
        )
        # diamonds for immediate current + future targets
        cv.drawMarker(
            display_frame,
            (current_path_x, rows - 10),
            (255, 0, 0),
            cv.MARKER_DIAMOND,
            11,
            3,
        )
        cv.drawMarker(
            display_frame,
            (future_path_x, rows - future_path_offset_px),
            (255, 0, 0),
            cv.MARKER_DIAMOND,
            11,
            3,
        )
        # line between current + future averaged targets
        cv.line(
            display_frame,
            (current_avg, rows),
            (future_avg, rows - future_path_offset_px),
            (0, 0, 255),
            3,
        )
        # path outputs
        # current/future lerp proportion
        cv.line(
            display_frame,
            (0, rows - lookahead_proportion),
            (cols, rows - lookahead_proportion),
            (255, 255, 0),
            2,
        )
        # current target output
        cv.line(
            display_frame,
            (current_target, 0),
            (current_target, rows),
            (255, 255, 0),
            2,
        )

        current_pid_px = int(current_pid_val.value * cols + cols / 2)
        cv.line(
            display_frame,
            (current_pid_px, 0),
            (current_pid_px, rows),
            (0, 255, 255),
            2,
        )
        # steering outputs
        steering_view_height = 90
        steering_view_x_offset = 5
        steering_view_y_offset = 5
        steering_view_width = 10
        steering_view_spacing = 50
        steering_view_y_center = steering_view_y_offset + int(steering_view_height / 2)
        left, right = steering_to_motor_vals(current_pid_val.value, path_lost)
        sv_right_px = int((steering_view_height / 2) * right)
        sv_left_px = int((steering_view_height / 2) * left)

        cv.rectangle(
            display_frame,
            (cols - steering_view_x_offset, steering_view_y_center),
            (
                cols - steering_view_x_offset - steering_view_width,
                steering_view_y_center - sv_right_px,
            ),
            (0, 255, 0) if sv_right_px >= 0 else (0, 0, 255),
            -1,
        )
        cv.rectangle(
            display_frame,
            (cols - steering_view_x_offset, steering_view_y_offset),
            (
                cols - steering_view_x_offset - steering_view_width,
                steering_view_y_offset + steering_view_height,
            ),
            (0, 0, 0),
            2,
        )
        cv.rectangle(
            display_frame,
            (
                cols - steering_view_x_offset - steering_view_spacing,
                steering_view_y_center,
            ),
            (
                cols
                - steering_view_x_offset
                - steering_view_width
                - steering_view_spacing,
                steering_view_y_center - sv_left_px,
            ),
            (0, 255, 0) if sv_left_px >= 0 else (0, 0, 255),
            -1,
        )
        cv.rectangle(
            display_frame,
            (
                cols - steering_view_x_offset - steering_view_spacing,
                steering_view_y_offset,
            ),
            (
                cols
                - steering_view_x_offset
                - steering_view_width
                - steering_view_spacing,
                steering_view_y_offset + steering_view_height,
            ),
            (0, 0, 0),
            2,
        )

        # pixel is in BGR!
        mouse_x = min(max(mouse_x, 0), cols - 1)
        mouse_y = min(max(mouse_y, 0), rows - 1)

        # show some pixel info on mouse hover for debugging
        rgbPixel = input_frame[mouse_y, mouse_x]
        hsvPixel = hsvImg[mouse_y, mouse_x]
        samplePixel = display_frame[mouse_y, mouse_x]

        # render debug info
        render_text(display_frame, txt, (5, 30))
        render_text(
            display_frame,
            f"X:{mouse_x:04d}, Y:{mouse_y:04d}",
            (5, 60),
        )
        render_text(
            display_frame,
            f"R{rgbPixel[2]:03d} G{rgbPixel[1]:03d} B{rgbPixel[0]:03d}",
            (5, 90),
        )
        render_text(
            display_frame,
            f"H{hsvPixel[0]:03d} S{hsvPixel[1]:03d} V{hsvPixel[2]:03d}",
            (5, 120),
        )
        render_text(display_frame, f"W:{cols} H:{rows}", (5, 180))
        render_text(
            display_frame,
            f"R{samplePixel[2]:03.0f} G{samplePixel[1]:03.0f} B{samplePixel[0]:03.0f}",
            (5, 210),
        )
        render_text(
            display_frame,
            f"ERR: {current_path_error_px:+04.0f} {future_path_error_px:+04.0f}",
            (5, 240),
        )
        if not (blu_detected or ylw_detected):
            render_text(
                display_frame,
                f"NO TRACK!",
                (5, 270),
            )
        elif blu_detected and not ylw_detected:
            render_text(
                display_frame,
                f"NO YELW!",
                (5, 270),
            )
        elif ylw_detected and not blu_detected:
            render_text(
                display_frame,
                f"NO BLUE!",
                (5, 270),
            )

        warn_text = ""
        if short_path_warn:
            warn_text = "NO FUTURE PATH"

        if len(warn_text) > 0:
            render_text(
                display_frame,
                f"WARN: {warn_text.upper()}",
                (5, 270),
                (0, 100, 255),
                (0, 0, 0),
            )

        if should_stop:
            render_text(
                display_frame,
                f"PATH LOSS TIMEOUT - STOP",
                (5, image_centre_y),
                (0, 0, 255),
                (0, 0, 0),
                2,
            )
        elif path_lost:
            render_text(
                display_frame,
                f"PATH LOST",
                (5, image_centre_y),
                (0, 180, 255),
                (0, 0, 0),
                2,
            )

        proc_dt = end_time - capture_time
        full_dt = end_time - start_time
        proc_time_ms = proc_dt * 1000
        full_time_ms = proc_dt * 1000
        fps = 1 / max(proc_dt, 0.00001)
        render_text(
            display_frame,
            # f"P,F:{proc_time_ms:03.0f},{full_time_ms:03.0f} FPS:{fps:03.0f}",
            f"dt:{full_time_ms:03.0f}ms FPS:{fps:03.0f}",
            (5, 150),
            col=((0, 0, 0) if fps > 25 else (0, 0, 255)),
        )

        # display_frame = cv.resize(
        #     display_frame, (int(cols * display_scale), int(rows * display_scale))
        # )
        if show:
            cv.imshow(window_title, display_frame)

    serial_should_stop.value = 1
    cv.destroyAllWindows()
    cap.release()
    serial_io_thread.join()
    ser.close()
