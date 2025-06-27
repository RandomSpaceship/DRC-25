# TERMINOLOGY:
# - junction: a point where 3 or more lines meet
# - termination: a point where 1 line ends
# - node: either a junction or a termination
# - line: A connection between two nodes
# - endpoint: A point on either end of a line which needs to be associated with a node

import config
import cv2 as cv
import numpy as np

local_region_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
averaging_kernel = local_region_kernel / np.sum(local_region_kernel)
junction_create_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))


def find_paths(mask):
    start_tick = cv.getTickCount()
    skeleton = cv.ximgproc.thinning(mask, thinningType=cv.ximgproc.THINNING_GUOHALL)
    rows, cols = mask.shape
    # dilate only the bottom row to ensure that there are junctions at the bottom of the image
    skeleton[rows - 1, :] = cv.dilate(
        skeleton[rows - 1, :], junction_create_kernel, iterations=1
    ).flatten()
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
    junction_endpoint_coords = [
        (int(x), int(y)) for (y, x) in np.argwhere(junction_endpoint_mask)
    ]
    termination_coords = [(int(x), int(y)) for (y, x) in np.argwhere(terminations)]
    endpoint_coords = junction_endpoint_coords + termination_coords

    # we're gonna use these a lot...
    junction_endpoint_count = len(junction_endpoint_coords)
    termination_count = len(termination_coords)
    endpoint_count = junction_endpoint_count + termination_count

    # INDEXING EXPLANATION:
    # Nodes are indexed from 0 to node_count - 1,
    # where 0 to junction_count - 1 are junctions,
    # and junction_endpoint_count to endpoint_count - 1 are terminations.
    # Endpoints are indexed from 0 to endpoint_count - 1,
    # where 0 to junction_endpoint_count - 1 are junction endpoints,
    # and junction_endpoint_count to endpoint_count - 1 are terminations.
    # Any "node index" is the former, and any "endpoint index" is the latter.

    endpoint_junction_map = {}
    junction_endpoint_map = {}
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
            dist = cv.pointPolygonTest(contour, endpoint, False)
            if dist < 0:
                # not inside junction contour, skip
                continue
            endpoint_junction_map[endpoint_idx] = junction_idx
            if junction_idx not in junction_endpoint_map:
                junction_endpoint_map[junction_idx] = []
            junction_endpoint_map[junction_idx].append(endpoint_idx)
            break

    # keep a list of remaining endpoints to associate with lines
    # this allows us to skip already-associated endpoints
    line_endpoint_indices_remaining = list(range(endpoint_count))
    # associate endpoints with their lines
    for line_idx, contour in enumerate(line_contours):
        endpoint_count = 0
        for endpoint_idx in line_endpoint_indices_remaining:
            endpoint = endpoint_coords[endpoint_idx]
            dist = cv.pointPolygonTest(contour, endpoint, False)
            if dist >= 0:
                line_endpoints[line_idx].append(endpoint_idx)
                endpoint_count += 1
            if endpoint_count == 2:
                # both endpoints found, no need to check further
                break

        # remove associated endpoint from the list of those remaining
        for endpoint_idx in line_endpoints[line_idx]:
            line_endpoint_indices_remaining.remove(endpoint_idx)

    # the root nodes of the node tree
    tree_roots = []
    # any node close to the bottom of the image may potentially be a root node,
    # however due to how the skeletonization works, there will almost always be 2 terminator nodes
    # right next to a junction node at the start of every path, and that makes things annoying,
    # so we just search junction roots only instead. We also dilate the bottom row of the skeletonised image
    # to ensure that there are junctions at the bottom of the image.

    # search for junction roots first, to ensure that potential loops will get overriden from terminator roots
    for current_idx, (x, y) in enumerate(junction_coords):
        if y >= rows - config.values["algorithm"]["path_root_y_threshold"]:
            tree_roots.append(current_idx)

    def find_line(endpoint_idx):
        """Find the line index for a given endpoint index."""
        for line_idx, endpoints in enumerate(line_endpoints):
            if len(endpoints) != 2:
                return None
            if endpoint_idx in endpoints:
                if endpoints[0] == endpoint_idx:
                    return (line_idx, endpoints[1])
                else:
                    return (line_idx, endpoints[0])
        return None

    # Construct the node tree, starting from the root nodes.
    # This uses the links (lines) between nodes, and the junction data to build a directed
    # (hopefully acyclic!) graph where each node can have multiple child nodes.
    tree = {}
    for tree_idx, root_idx in enumerate(tree_roots):
        tree[root_idx] = []
        leaf_indices = []
        if root_idx < junction_endpoint_count:
            if root_idx not in junction_endpoint_map:
                continue
            leaf_indices = [(root_idx, idx) for idx in junction_endpoint_map[root_idx]]
        else:
            leaf_indices = [(root_idx, root_idx)]
        while leaf_indices:
            (prev_idx, current_node_idx) = leaf_indices.pop(0)

            line_data = find_line(current_node_idx)
            if line_data is None:
                # print("Error: No line found for endpoint", current_node_idx)
                continue
            line_idx, other_endpoint_idx = line_data

            # if it's a junction endpoint...
            if other_endpoint_idx < junction_endpoint_count:
                junction_idx = endpoint_junction_map[other_endpoint_idx]
                if junction_idx is None:
                    # print("Error: No junction found for endpoint", current_node_idx)
                    continue
                other_nodes = junction_endpoint_map[junction_idx]
                if other_endpoint_idx in other_nodes:
                    other_nodes.remove(other_endpoint_idx)
                tree[junction_idx] = []  # no loops allowed!
                # if junction_idx in tree and prev_idx in tree[junction_idx]:
                #     tree[junction_idx].remove((prev_idx, line_idx))

                tree[prev_idx].append((junction_idx, line_idx))
                leaf_indices.extend([(junction_idx, idx) for idx in other_nodes])
            # otherwise, it's a termination
            else:
                tree[other_endpoint_idx] = []  # ensure that there are no loops
                tree[prev_idx].append((other_endpoint_idx, line_idx))

    end_tick = cv.getTickCount()
    dt = (end_tick - start_tick) / cv.getTickFrequency()

    return {
        "tree": tree,
        "roots": tree_roots,
        "junctions": junction_coords,
        "junction_endpoint_count": junction_endpoint_count,
        "terminations": termination_coords,
        "line_lengths": line_lengths,
        "proc_time": dt,
    }
