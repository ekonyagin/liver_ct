import numpy as np
from dpipe.checks import add_check_bool, add_check_shapes
from dpipe.im import describe_connected_components
from dpipe.im.metrics import iou


@add_check_bool
@add_check_shapes
def match_gt_and_pred(gt, pred, metric=iou, metric_th=0.5, one_to_many=False, return_not_matched=False):
    """
    If one_to_many is True the function matches one GT's CC to many predicted CCs, else matches it one to one.
    """
    labeled_gt, labels_gt, volumes_gt = describe_connected_components(gt)
    labeled_pred, labels_pred, volumes_pred = describe_connected_components(pred)

    labels_gt, labels_pred = list(labels_gt), list(labels_pred)
    res_gt, res_pred = np.zeros_like(gt).astype(int), np.zeros_like(pred).astype(int)

    i = 1
    for gt_label in labels_gt.copy():
        temp_res_gt = labeled_gt == gt_label
        metric_values = [metric(temp_res_gt, labeled_pred==pred_label) for pred_label in labels_pred]

        if not one_to_many:
            good_iou_pred_labels = [labels_pred[np.argmax(metric_values)]] if np.max(metric_values) >= metric_th else []
        else:
            good_iou_pred_labels = np.array(labels_pred)[np.array(metric_values) >= metric_th]

        for pred_label in good_iou_pred_labels:
            labels_pred.remove(pred_label)
            res_gt[temp_res_gt] = i
            res_pred[labeled_pred==pred_label] = i

        if len(good_iou_pred_labels):
            labels_gt.remove(gt_label)
            i+=1

    if return_not_matched:
        for j, gt_label in enumerate(labels_gt, 1):
            res_gt[labeled_gt == gt_label] = -1*j

        for j, pred_label in enumerate(labels_pred, 1):
            res_pred[labeled_pred == pred_label] = -1*j

    tp, fn, fp = i-1, len(labels_gt), len(labels_pred)

    return res_gt, res_pred, tp, fn, fp


# --- multilabel stuff

def probamap2segm(x: np.ndarray, axis=1) -> np.ndarray:
    if x.shape[axis] == 1:
        ans = (x > 0.5).take(0, axis=axis).astype(int)
        return ans
    else:
        ans = np.argmax(x, axis=axis)#[None].swapaxes(1, 0)
        return ans


def probamap2segm_score(score):
    def wrapped(y: np.ndarray, x: np.ndarray, *args, **kwargs):
        x = probamap2segm(x)
        return score(y=y, x=x, *args, **kwargs)

    return wrapped


def multichannel_score(y: np.ndarray, x: np.ndarray, score, multilabel_mask_splitter) -> list:
    x = probamap2segm(x, axis=0)
    return [score(x=x_, y=y_) for x_, y_ in zip(*map(multilabel_mask_splitter, [x, y]))]
