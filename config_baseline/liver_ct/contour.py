from itertools import product

from skimage.draw import line, polygon2mask
import numpy as np


def len_contour(contour):
    return np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()


def check_orientation(point: np.array, contour_slice):
    for line_ in zip(contour_slice[:-1], contour_slice[1:]):
        vec_ = line_[1] - line_[0]
        vec_ = vec_[::-1] * [-1, 1]
        p_ = point - line_[0]

        if (vec_ * p_).sum() < 0:
            return 0

    return 1


def plot_curve(curve, shape):
    shape = shape[:2]

    m = np.zeros(shape, bool)
    curve = np.round(curve).astype(int)
    for a, b in zip(curve[:-1], curve[1:]):
        rr, cc = line(a[0], a[1], b[0], b[1])
        m[rr, cc] = True
    return m


def extrapolate_segment_to_border(segment, shape):
    """
    Returns border point's coordinates
    """
    shape = np.array(shape) - 1
    point, vector = segment[0], segment[1] - segment[0]
    vector = vector / np.linalg.norm(vector)
    borders = [
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [shape, [-1, 0]],
        [shape, [0, -1]]]

    borders = np.array(borders).astype(float)

    f_ = lambda x, y: x[0] * y[1] - x[1] * y[0]
    k_min = np.inf
    for b_point, b_vector in borders:
        if np.abs(f_(b_vector, vector)) > 1e-9:
            k = (f_(point, b_vector) - f_(b_point, b_vector)) / f_(b_vector, vector)
            if 0 < k < k_min: k_min = k

    return np.clip(point+vector*k_min, 0, shape)


def extrapolate_first_and_last_segments_to_border(curve: np.array, shape):
    shape = shape[:2]
    cs = np.insert(curve, 0, extrapolate_segment_to_border(curve[:2][::-1], shape), axis=0)
    cs = np.insert(cs, len(cs), extrapolate_segment_to_border(curve[-2:], shape), axis=0)

    return cs


def extrapolate_first_and_last_segments(curve: np.array, extrapol_len=20, shape=None):
    delta_vec = curve[0] - curve[1]
    delta_vec = delta_vec / np.linalg.norm(delta_vec)
    cs = np.insert(curve, 0, curve[0] + extrapol_len * delta_vec, axis=0)

    delta_vec = curve[-1] - curve[-2]
    delta_vec = delta_vec / np.linalg.norm(delta_vec)
    cs = np.append(cs, [curve[-1] + extrapol_len * delta_vec], axis=0)

    return cs


def find_segments_intersection(segment1, segment2):
    """
    Returns intersection point coordinates.
    """
    # We use segments described by a dot and a vector: (A, alpha), (B, beta)
    alpha, beta = np.diff(segment1, axis=0)[0], np.diff(segment2, axis=0)[0]
    A, B = segment1[0], segment2[0]

    f_ = lambda x, y: x[0] * y[1] - x[1] * y[0]

    # check if segments are collinear
    if np.abs(f_(beta, alpha)) > 1e-9:
        k = (f_(A, beta) - f_(B, beta)) / f_(beta, alpha)
        m = (f_(B, alpha) - f_(A, alpha)) / f_(alpha, beta)
        # check if intersection point is in a segment
        if 0 <= k <= 1 and 0 <= m <= 1:
            return A + k * alpha


def orient_I(contour_slice):
    a, b = contour_slice[0], contour_slice[-1]
    if a[0] < b[0]:
        contour_slice = contour_slice[::-1]

    return contour_slice


def orient_Line(contour_slice, shape):
    """
    From a centre to out
    """
    shape_2d = np.array(shape[:2])
    a, b = contour_slice[0], contour_slice[-1]
    centre = shape_2d / 2

    if np.linalg.norm(b - centre) < np.linalg.norm(a - centre):
        contour_slice = contour_slice[::-1]

    return contour_slice


# --- useful utils
def close_contour(contour):
    return np.insert(contour, len(contour), contour[0], axis=0)


def get_segments(contour):
    return zip(contour[:-1], contour[1:])


# --- chopping
def chop_segment(segment: np.array, size):
    assert size > 1e-4
    segment = np.array(segment).astype(float)
    start, end = segment
    vec_ = end - start

    segment_len = np.linalg.norm(vec_)
    if segment_len <= size + 1e-4:
        return segment
    vec_ = vec_ / segment_len * size

    return np.insert(segment, 1, [start + i * vec_ for i in range(1, np.floor(segment_len / size).astype(int) + 1)],
                     axis=0)


def chop_curve(contour: np.array, size):
    """
    Chop the curve. Each segment of the curve will have length <= size.
    """
    return np.vstack([chop_segment(segment, size)[:-1]
                      for segment in get_segments(contour)] + [contour[-1]])


# ---
def find_nearest(point, cloud):
    return cloud[np.argmin(np.linalg.norm(cloud - point, axis=1))]


# --- split contour by a curve
def get_subcontours(contour, a: int, b: int):
    """
    Split the contour by two indexes.
    Each contour starts from contour[a] and ends with contour[b].

    """
    a_, b_ = min(a, b), max(a, b)

    c_1 = contour[a_:b_ + 1]
    c_2 = np.vstack([contour[b_:], contour[:a_ + 1]])[::-1]

    if a > b:
        c_1 = c_1[::-1]
        c_2 = c_2[::-1]

    return c_1, c_2


def get_angle(vec_1, vec_2):
    """
    Get oriented angle between vec_1 and vec_2. Positive orientation - counterclockwise
    """
    vec_1 = vec_1 / np.linalg.norm(vec_1)
    vec_2 = vec_2 / np.linalg.norm(vec_2)
    sign_ = 1 if (vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]) > 0 else - 1

    return np.arccos(np.clip(np.sum(vec_1 * vec_2), -1, 1)) * sign_


def check_if_inside(point, contour):
    """
    Calculate sum of angles between [contour's point - point] vectors.
    The sum is zero for outside point, and +-2pi (consistent with the orientation of the contour) for inside point
    """
    return np.abs(np.sum([get_angle(*(point - segment))
                          for segment in get_segments(close_contour(contour))])) > 0.1


def get_segment_contour_intersections(segment, contour):
    l = []
    for contour_segment in get_segments(close_contour(contour)):
        res = find_segments_intersection(segment, contour_segment)
        if res is not None:
            l.append(res)

    if l:
        return np.array(l)


def get_curve_contour_intersections(curve, contour):
    l = []
    for curve_segment in get_segments(curve):
        res = get_segment_contour_intersections(curve_segment, contour)
        if res is not None:
            l.extend(res)

    if l:
        return np.array(l)


def cut_curve_outside(curve, contour):
    """
    Returns subcurve inside the contour. Curve should have exactly one subcurve inside.
    The function also appends intersection points to the curve
    ---
    ATTENTION!
    If all vertex of the curve are inside the contour, curve will not be changed,
    but it this case some subcurve may be outside the contour
    """
    # get points inside the contour
    l = [check_if_inside(p, contour) for p in curve]
    a, b = np.argmax(l), len(l) - np.argmax(l[::-1])

    # if no vertex inside the contour we replace it by intersection points
    if not np.any(l):
        res = get_curve_contour_intersections(curve, contour)
        # it should have exactly 2 intersection points
        # it should be a segment with both end points outside the contour
        if res is not None:
            if len(res) != 2: raise ValueError('No vertex inside, the curve should have exactly 2 intersection points')
            return res

    # whole subcurve should be inside the contour
    if not np.all(l[a:b]): raise ValueError('Whole subcurve should be inside the contour')
    res = curve[a:b]

    # append intersection points to the curve
    if a != 0:
        point = get_segment_contour_intersections(curve[a - 1:a + 1], contour)
        if len(point) != 1: raise ValueError('Borderline segment should have exactly one intersection point')
        res = np.insert(res, 0, point[0], axis=0)

    if b != len(curve):
        point = get_segment_contour_intersections(curve[b - 1:b + 1], contour)
        if len(point) != 1: raise ValueError('Borderline segment should have exactly one intersection point')
        res = np.insert(res, len(res), point[0], axis=0)

    return res


def split_contour_by_curve(curve, contour, size=1):
    """
    Returns two contours from the contour spited by the curve.
    The curve should have exactly one subcurve inside (see cut_curve_outside).
    The first part of both returned contour is the inside subcurve.
    """
    curve = cut_curve_outside(curve, contour)

    cc = close_contour(contour)
    # [:-1] removes duplicated first point after close_contour
    cc = chop_curve(cc, size)[:-1]

    a = np.argmin(np.linalg.norm(cc - curve[0], axis=1))
    b = np.argmin(np.linalg.norm(cc - curve[-1], axis=1))

    c1, c2 = get_subcontours(cc, a, b)
    if c1.mean(axis=0)[1] > c2.mean(axis=0)[1]:
        c1, c2 = c2, c1

    return np.insert(curve, len(curve), c1[::-1], axis=0), np.insert(curve, len(curve), c2[::-1], axis=0)


def get_zones_for_slice_by_curves(contours, shape, size=1, curves_values=[]):
    liver_slice, *curves = contours

    shape = shape[:2]
    mask = np.zeros(shape)
    if not liver_slice.any():
        return mask

    contour = liver_slice.copy()
    last_hit = 0
    for i, curve in enumerate(curves, 1):
        if not curve.any():
            continue
        # if we have not correct curve split_contour_by_curve returns ValueError (see cut_curve_outside)
        try:
            contour, zone = split_contour_by_curve(curve, contour, size)

            # check if we swap contour and zone
            # angle magic - we want to cut zones in clockwise order
            contour_centre, zone_centre = contour.mean(axis=0), zone.mean(axis=0)
            contour_angle = get_angle(shape-contour_centre, np.array([0, -1]))
            zone_angle = get_angle(shape - zone_centre, np.array([0, -1]))
            if zone_angle > contour_angle:
                contour, zone = zone, contour

        except ValueError:
            continue
        last_hit = i
        temp_mask = polygon2mask(shape, zone)
        mask[temp_mask] = i

    temp_mask = polygon2mask(shape, contour)
    mask[temp_mask] = last_hit+1

    # TODO: add pixels in plot_curve to returned mask
    if curves_values:
        for n, c in zip(curves_values, contours):
            mask[plot_curve(c, shape)] = n

    return mask


# --- only lines annotation stuff

def intersect_line_and_image(segment, shape):
    """
    Returns border point's coordinates or []
    """
    shape = np.array(shape) - 1
    point, vector = segment[0], segment[1] - segment[0]
    vector = vector / np.linalg.norm(vector)
    borders = [
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [shape, [-1, 0]],
        [shape, [0, -1]]]

    borders = np.array(borders).astype(float)

    f_ = lambda x, y: x[0] * y[1] - x[1] * y[0]

    res = []
    for b_point, b_vector in borders:
        if np.abs(f_(b_vector, vector)) > 1e-9:
            m = (f_(b_point, vector) - f_(point, vector)) / f_(vector, b_vector)
            res.append(m)
        else:
            res.append(-np.inf)
    res = np.array(res)
    idx = (0 <= res) & (res <= np.hstack([shape, shape]))

    if not idx.any():
        return []

    points = (borders[:, 0] + borders[:, 1]*res[:, None])[idx]
    points = np.unique(points, axis=0)
    assert len(points) == 2

    return points


def get_zones_for_slice_by_lines(curves, shape, curves_values=[]):
    """
    curves : slices of ['Left', 'Middle','Right']
    """
    shape = shape[:2]
    mask = np.zeros(shape)

    bounds = [[0,0],
              [0, shape[1]],
              shape,
              [shape[0], 0]]
    bounds = np.array(bounds)

    for n, curve in zip([1, 10, 100], curves):
        if not curve.any():
            raise ValueError('Got empty curve')
            # continue

        # extrapolate line to borders
        curve = intersect_line_and_image(curve, shape)

        contour, zone = split_contour_by_curve(curve, bounds, 1)
        # check if we swap contour and zone
        # angle magic - we want to cut zones in clockwise order
        contour_centre, zone_centre = contour.mean(axis=0), zone.mean(axis=0)
        contour_angle = get_angle(shape-contour_centre, np.array([0, -1]))
        zone_angle = get_angle(shape - zone_centre, np.array([0, -1]))
        if zone_angle > contour_angle:
            contour, zone = zone, contour

        mask += n*polygon2mask(shape, zone)

    for i, n in enumerate([111, 110, 100, 0], 1):
        mask[mask==n] = i

    # TODO: add pixels in plot_curve to returned mask
    if curves_values:
        for n, c in zip(curves_values, curve):
            c = intersect_line_and_image(c, shape)
            mask[plot_curve(c, shape)] = n

    return mask
