import config
import cv2 as cv
import math
import numpy as np

# import hardware

print("\r\n\r\nDRC Main Launch File")

print(config.values)

# hardware_api = hardware.HardwareAPI()

window_title = "Pathfinder"

img = cv.imread("paths.png")
rows, cols, channels = img.shape

cv.namedWindow(window_title, cv.WINDOW_GUI_NORMAL)
cv.resizeWindow(window_title, int(cols), int(rows))

while True:
    key = cv.waitKey(1)
    if key == ord("-"):
        break

    img = cv.imread("paths3.png")

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # lines = cv.HoughLines(img_gray, 1, np.pi / 180, 150, None, 0, 0)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         cv.line(img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    # linesP = cv.HoughLinesP(img_gray, 1, 30 * np.pi / 180, 80, None, 0, int(rows * 0.1))
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    # find contours
    contours, heierarchy = cv.findContours(
        img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    # disp = np.zeros((rows, cols, 3), np.uint8)
    skeleton = cv.ximgproc.thinning(img_gray, thinningType=cv.ximgproc.THINNING_GUOHALL)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    kernel = kernel / np.sum(kernel)
    blur = cv.filter2D(skeleton, -1, kernel)
    # averaging blur
    # blur_skel = cv.GaussianBlur(skeleton, (5, 5), 0)
    _, junctions = cv.threshold(blur, 60, 255, cv.THRESH_BINARY)

    # with junctions found:
    # - strip them out of the skeletonised image
    # -

    (height, width) = skeleton.shape
    pointsMask = np.zeros((height, width, 1), np.uint8)

    # 0.15: too low
    # 0.2: endpoints, junctions
    # 0.4: most junctions (shallow ones fail)
    corners = cv.goodFeaturesToTrack(skeleton, 25, 0.3, 10)

    disp = cv.cvtColor(junctions, cv.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        # cv.circle(disp, (int(x), int(y)), 5, (255, 100, 0), -1)

    for contour in contours:
        # approx = cv.approxPolyDP(contour, 0.05 * cv.arcLength(contour, True), True)
        # print(f"len {cv.arcLength(contour, True)}")
        # sides = math.floor(cv.arcLength(contour, True) * 0.05)
        # if sides > len(contour) / 2:

        # convex hull not having enough points?
        approx = cv.convexHull(contour, returnPoints=True)

        # cv.drawContours(disp, [contour], 0, (0, 0, 255), 3)
        # cv.drawContours(disp, [approx], 0, (0, 255, 0), 3)
        x, y, w, h = cv.boundingRect(contour)
        # cv.rectangle(disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # https://stackoverflow.com/questions/72164740/how-to-find-the-junction-points-or-segments-in-a-skeletonized-image-python-openc
    # blur = cv.GaussianBlur(disp, (3,3), 0)
    # thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # # Find horizonal lines
    # horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,1))
    # horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=1)

    # # Find vertical lines
    # vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,5))
    # vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=1)

    # # Find joint intersections then the centroid of each joint
    # joints = cv.bitwise_and(horizontal, vertical)
    # cnts = cv.findContours(joints, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     # Find centroid and draw center point
    #     x,y,w,h = cv.boundingRect(c)
    #     centroid, coord, area = cv.minAreaRect(c)
    #     cx, cy = int(centroid[0]), int(centroid[1])
    #     cv.circle(disp, (cx, cy), 5, (36,255,12), -1)

    # # Find endpoints
    # corners = cv.goodFeaturesToTrack(thresh, 5, 0.5, 10)
    # corners = np.int0(corners)
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv.circle(disp, (x, y), 5, (255,100,0), -1)

    # cv.imshow('thresh', thresh)
    # cv.imshow('joints', joints)
    # cv.imshow('horizontal', horizontal)
    # cv.imshow('vertical', vertical)
    # cv.imshow('image', disp)
    # cv.waitKey()

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
