import numpy as np
from neurodata.contours import contours_to_mask, interpolate_dict
from skimage.morphology import binary_opening, binary_closing, disk, flood_fill
from sklearn.decomposition import PCA

from .contour import extrapolate_first_and_last_segments_to_border, plot_curve, intersect_line_and_image


def format_annotator_json(ann_json):
    d = {}
    for i in ann_json:
        d[i['name']] = i

    return d


# TODO: add pixels in plot_curve to returned mask
def load_horizontal_mask(ann_d, liver_mask, verbose_curve=True):
    shape = liver_mask.shape
    curve_dict = ann_d['Horizontal']['data']
    l = liver_mask.any(axis=(1,2))
    liver_start, liver_stop = np.argmax(l), len(l) - np.argmax(l[::-1]) - 1
    curve_start, curve_stop = np.array(sorted(map(int, curve_dict.keys())))[[0, -1]]
    curve_dict[str(liver_start)] = curve_dict[str(curve_start)]
    curve_dict[str(liver_stop)] = curve_dict[str(curve_stop)]

    curve_slices = interpolate_dict(curve_dict, shape[0], closed=False)
    mask = []
    for curve_slice, liver_slice in zip(curve_slices, liver_mask):
        m_ = np.zeros_like(liver_slice)
        if not curve_slice.any():
            mask.append(m_)
            continue
        curve_slice = extrapolate_first_and_last_segments_to_border(curve_slice, liver_slice.shape)
        m_ = flood_fill(m_+2*plot_curve(curve_slice, liver_slice.shape), (0,0), new_value=1, connectivity=1) == 1
        # TODO: get zones for all image, not only for a liver
        m_ = (m_+1) * liver_slice

        if verbose_curve:
            m_[plot_curve(curve_slice, liver_slice.shape)] = -1

        mask.append(m_)

    return np.stack(mask)


# --- plane approximation

def fit_plane_by_curve_dict(curve_dict):
    curve_array = [np.hstack([value, len(value) * [[int(key)]]]) for key, value in curve_dict.items()]
    curve_array = np.vstack(curve_array)

    solver = PCA(n_components=2, random_state=42)
    solver.fit(curve_array)

    return solver.components_, solver.mean_


def get_fitted_plane_slices(curve_dict, shape):
    (alpha, beta), m = fit_plane_by_curve_dict(curve_dict)

    res = []
    for z in range(shape[-1]):
        a = (z - m[-1]) / alpha[-1]
        b = (z - m[-1]) / beta[-1]
        segment = np.array([m + a * alpha, m + b * beta])[..., :-1]

        res.append(intersect_line_and_image(segment, shape[:-1]))

    return res


def get_distances_to_fitted_plane(curve_dict):
    curve_array = [np.hstack([value, len(value) * [[int(key)]]]) for key, value in curve_dict.items()]
    curve_array = np.vstack(curve_array)

    (alpha, beta), m = fit_plane_by_curve_dict(curve_dict)
    n = np.cross(alpha,beta)

    return np.dot(curve_array - m, n)


# --- only lines annotation stuff
