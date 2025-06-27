# TERMINOLOGY:
# - junction: a point where 3 or more lines meet
# - termination: a point where 1 line ends
# - node: either a junction or a termination
# - line: A connection between two nodes
# - endpoint: A point on either end of a line which needs to be associated with a node

import config
import cv2 as cv
import math
import networkx as nx
import numpy as np
from enum import Enum

# import hardware

print("\r\n\r\nDRC Main Launch File")

print(config.values)

# hardware_api = hardware.HardwareAPI()

window_title = "Pathfinder"

img = cv.imread("path.png")
rows, cols, channels = img.shape

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


local_region_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
averaging_kernel = local_region_kernel / np.sum(local_region_kernel)

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

    start_tick = cv.getTickCount()

    skeleton = cv.ximgproc.thinning(
        input_mask, thinningType=cv.ximgproc.THINNING_GUOHALL
    )
    # averaging blur (used to count local pixels)
    averaged_mask = cv.filter2D(skeleton, -1, averaging_kernel)
    # All junctions are guaranteed to have > 3 pixels in the 8-neighborhood,
    # and as such all junctions are guaranteed to have at least 4 pixels in the 3x3 neighborhood.
    # For certain line geometries, this may have false positives, but this isn't really a problem
    junction_threshold = int(255 * 3.5 / 9)
    _, junctions = cv.threshold(
        averaged_mask, junction_threshold, 255, cv.THRESH_BINARY
    )
    # terminations are guaranteed to only have 1 pixel in the 8-neighborhood,
    # and therefore 2 pixels in the 3x3 neighborhood.
    termination_threshold = int(255 * 2.5 / 9)
    _, terminations = cv.threshold(
        averaged_mask, termination_threshold, 255, cv.THRESH_BINARY_INV
    )
    # However, this check also catches the blurred edges of the lines,
    # along with the black sections of the mask,
    # so only pixels that were originally part of the skeleton
    # will be valid. This method will guarantee single-pixel terminations,
    # although they may overlap with junctions in some cases (which will need to be removed).
    # Each termination is also guaranteed to be a line endpoint.
    terminations = cv.bitwise_and(terminations, skeleton)

    # split_skeleton is the skeleton with junctions removed.
    # This splits the skeleton into individual lines.
    split_skeleton = skeleton.copy()
    split_skeleton[junctions > 0] = 0

    # expanding the junctions with the 8-neighborhood ensures that
    # the junctions will overlap with lines by exactly one pixel
    expanded_junctions = cv.dilate(junctions, local_region_kernel, iterations=1)

    # we also need to remove line segments that fully overlap with the expanded junctions,
    # as they will not be linked correctly.
    # To do so, we strip out the expanded junctions
    # (which should only take one pixel from the end of a line segment)...
    expanded_split_skeleton = cv.bitwise_and(
        split_skeleton, cv.bitwise_not(expanded_junctions)
    )
    # ... dilate the split skeleton (thus restoring the removed pixel), ...
    expanded_split_skeleton = cv.dilate(
        expanded_split_skeleton, local_region_kernel, iterations=1
    )
    # ... and then binary-AND it with the original split skeleton to ensure
    # that the skeleton is still 1 pixel wide.
    split_skeleton = cv.bitwise_and(split_skeleton, expanded_split_skeleton)
    # This will keep any lines which are not fully covered by junctions,
    # as the removed pixels will just be restored by the dilation, but will remove any lines
    # which are fully covered by junctions as *all* pixels will be removed in the original stripping.

    # TODO: Investigate contour (line, junction) intersection tests
    line_contours, _ = cv.findContours(
        split_skeleton, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    line_count = len(line_contours)

    junction_contours, _ = cv.findContours(
        expanded_junctions, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    junction_count = len(junction_contours)

    # remove terminations overlapping with junctions as they're invalid
    terminations[expanded_junctions > 0] = 0

    junction_endpoint_mask = cv.bitwise_and(split_skeleton, expanded_junctions)
    # find the coordinates of all line endpoints (terminations and junctions)
    junction_endpoint_coords = np.argwhere(junction_endpoint_mask)
    termination_coords = np.argwhere(terminations)
    endpoint_coords = np.vstack((junction_endpoint_coords, termination_coords))

    # we're gonna use these a lot...
    junction_endpoint_count = len(junction_endpoint_coords)
    termination_count = len(termination_coords)
    endpoint_count = junction_endpoint_count + termination_count
    node_count = junction_count + termination_count

    # INDEXING EXPLANATION:
    # Nodes are indexed from 0 to node_count - 1,
    # where 0 to junction_count - 1 are junctions,
    # and junction_count to node_count - 1 are terminations.
    # Endpoints are indexed from 0 to endpoint_count - 1,
    # where 0 to junction_endpoint_count - 1 are junction endpoints,
    # and junction_endpoint_count to endpoint_count - 1 are terminations.
    # Any "node index" is the former, and any "endpoint index" is the latter.

    endpoint_junction_map = {}
    # list of endpoints for each line
    line_endpoints = [[] for _ in range(line_count)]

    def get_contour_center(contour):
        """Calculate the center of a contour."""
        M = cv.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    junction_coords = [get_contour_center(contour) for contour in junction_contours]
    # arc length = total perimeter, so therefore length will be half of that since the widths at the end are 1 pixel
    # and as such don't contribute (significantly)
    line_lengths = [cv.arcLength(contour, True) // 2 for contour in line_contours]

    # associate junction endpoints with their junction
    for endpoint_idx, endpoint in enumerate(junction_endpoint_coords):
        for junction_idx, contour in enumerate(junction_contours):
            dist = cv.pointPolygonTest(
                contour, (int(endpoint[1]), int(endpoint[0])), False
            )
            if dist < 0:
                # not inside junction contour, skip
                continue
            endpoint_junction_map[endpoint_idx] = junction_idx
            break

    # keep a list of remaining endpoints to associate with lines
    # this allows us to skip already-associated endpoints
    line_endpoint_indices_remaining = list(range(endpoint_count))
    # associate endpoints with their lines
    for line_idx, contour in enumerate(line_contours):
        endpoint_count = 0
        clear_list = []
        for endpoint_idx in line_endpoint_indices_remaining:
            endpoint = endpoint_coords[endpoint_idx]
            dist = cv.pointPolygonTest(
                contour, (int(endpoint[1]), int(endpoint[0])), False
            )
            node_idx = endpoint_idx
            if endpoint_idx < junction_endpoint_count:
                node_idx = endpoint_junction_map[endpoint_idx]
            if dist >= 0:
                line_endpoints[line_idx].append(node_idx)
                clear_list.append(endpoint_idx)
                endpoint_count += 1
            if endpoint_count == 2:
                # both endpoints found, no need to check further
                break

        # remove associated endpoint from the list of those remaining
        for endpoint_idx in clear_list:
            line_endpoint_indices_remaining.remove(endpoint_idx)

    G = nx.DiGraph()
    for idx, (x, y) in enumerate(junction_coords):
        G.add_node(idx, pos=(int(x), int(y)))
    for idx, (y, x) in enumerate(termination_coords):
        G.add_node(idx + junction_endpoint_count, pos=(int(x), int(y)))

    G.add_edges_from(
        [
            (pair[0], pair[1], {"length": line_lengths[line_idx]})
            for line_idx, pair in enumerate(line_endpoints)
            if len(pair) == 2
        ]
    )

    # the root nodes of the node tree
    tree_termination_roots = []
    tree_junction_roots = []
    # any node close to the bottom of the image is a root node

    for current_idx, (y, x) in enumerate(termination_coords):
        if y >= rows - 5:
            tree_termination_roots.append(current_idx + junction_endpoint_count)

    for current_idx, (x, y) in enumerate(junction_coords):
        if y >= rows - 5:
            tree_junction_roots.append(current_idx)

    def swap_edge(u, v):
        data = G[v][u]
        G.remove_edge(v, u)
        G.add_edge(u, v, **data)

    for root in tree_junction_roots:
        edges = [x for x in G.in_edges(root)]
        for edge in edges:
            swap_edge(root, edge[0])

    invalid_termination_roots = []
    for root in tree_termination_roots:
        edges = [x for x in G.in_edges(root)]
        for edge in edges:
            if edge[0] not in tree_junction_roots:
                swap_edge(root, edge[0])
            else:
                invalid_termination_roots.append(root)

    for root in invalid_termination_roots:
        tree_termination_roots.remove(root)

    roots = tree_junction_roots + tree_termination_roots

    def propagate_outward(graph, prev_node, this_node):
        in_edges = [x for x in G.in_edges(this_node)]
        for next_node, _ in in_edges:
            if next_node != prev_node:
                swap_edge(this_node, next_node)
        out_edges = [x for x in G.out_edges(this_node)]
        for _, next_node in out_edges:
            propagate_outward(graph, this_node, next_node)

    for root in roots:
        propagate_outward(G, -1, root)

    disp = None
    match shown_image:
        case ShownImage.INPUT:
            disp = img.copy()
        case ShownImage.SKELETON:
            disp = cv.cvtColor(skeleton, cv.COLOR_GRAY2BGR)
        case ShownImage.JUNCTIONS:
            disp = cv.cvtColor(junctions, cv.COLOR_GRAY2BGR)
        case ShownImage.FILTERED_JUNCTIONS:
            disp = cv.cvtColor(expanded_junctions, cv.COLOR_GRAY2BGR)
        case ShownImage.TERMINATIONS:
            disp = cv.cvtColor(terminations, cv.COLOR_GRAY2BGR)
        case ShownImage.FILTERED_TERMINATIONS:
            disp = cv.cvtColor(terminations, cv.COLOR_GRAY2BGR)
        case ShownImage.BLUR:
            disp = cv.cvtColor(averaged_mask, cv.COLOR_GRAY2BGR)
        case ShownImage.SKELETON_MINUS_JUNCTIONS:
            disp = cv.cvtColor(split_skeleton, cv.COLOR_GRAY2BGR)

    def draw_node(node_idx):
        node = G.nodes[node_idx]
        for edge in G.out_edges(node_idx):
            pos = node["pos"]
            child_pos = G.nodes[edge[1]]["pos"]
            cv.arrowedLine(
                disp,
                (int(pos[0]), int(pos[1])),
                (int(child_pos[0]), int(child_pos[1])),
                (255, 255, 0),
                1,
            )
            if draw_terminations:
                cv.putText(
                    disp,
                    str(G.edges[edge]["length"]),
                    (
                        (int(pos[0]) + int(child_pos[0])) // 2,
                        (int(pos[1]) + int(child_pos[1])) // 2,
                    ),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                )
            draw_node(edge[1])

    if draw_junctions:
        for node in tree_junction_roots:
            draw_node(node)
        for node in tree_termination_roots:
            draw_node(node)

    if draw_junctions:
        disp[expanded_junctions > 0] = (0, 255, 0)
    if draw_terminations:
        disp[terminations > 0] = (255, 0, 255)  # endpoints
    end_tick = cv.getTickCount()
    dt = (end_tick - start_tick) / cv.getTickFrequency()
    print(dt)

    cv.imshow(window_title, disp)
cv.destroyAllWindows()
