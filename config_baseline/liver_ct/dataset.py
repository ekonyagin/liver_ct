import numpy as np
from scipy.stats import rv_discrete
from dpipe.medim.shape_ops import zoom
#from liver_ct.liver_localization import SelectSlice

def SelectSlice(pred,
                threshold_x=100,
                threshold_y=100, 
                threshold_z=100, 
                epsilon=12):
    """
    input: img: 3D numpy array of CT binary mask,
            threshold_i: sensitivity of the mask to number of True pixels
            epsilon: offset in pixels (how many pixels from liver computed boundaries we take)
    output: 3D numpy array containing liver
    """
    epsilon=10

    epsilon_upper = int(0.7*epsilon)

    sum_z = pred.sum(axis=0).sum(axis=0)
    min_z = np.argwhere(sum_z>100).min()
    max_z = np.argwhere(sum_z>100).max()
    min_z = np.maximum(min_z-epsilon,0)
    max_z = np.minimum(max_z+epsilon, len(sum_z))

    return min_z, max_z

def resize_image(image, factor, axes_to_scale=np.array([0,1,2])):
    return zoom(image, scale_factor=factor[axes_to_scale],
                axes=axes_to_scale, fill_value=np.min)


def resize_binary_mask(mask, factor, axes_to_scale=np.array([0,1,2])):
    return resize_image(mask, factor, axes_to_scale) >= 0.5


def get_patch_distribution(y, delta=50, plus=1):
    """
    Sampling near mask's z upper and lower bounds
    Can be used only with 1d z-patch sampling
    :param y:
    :param delta:
    :return:
    """
    
    # calculate our custom distribution
    y_1d = y.any(axis=(0,1))
    start, stop = np.argmax(y_1d), len(y_1d) - np.argmax(y_1d[::-1])

    pdf = np.ones(len(y_1d))
    pdf[max(0, start - delta):start] += plus
    pdf[max(0, stop - delta):stop] += plus

    # make Callable sampling function
    def distribution(shape):
        z_max = shape[0]
        dist = rv_discrete(a=0, b=z_max, values=(list(range(z_max)), pdf[:z_max] / pdf[:z_max].sum()))()

        return dist.rvs([1])

    return distribution


# --- multilabel stuff

def one2many(mask, n=None):
    if n is None:
        n = int(mask.max())
    return np.stack([mask == i for i in range(1, n+1)])


def many2one(mask):
    m = np.zeros(mask.shape[1:])
    for i, mod in enumerate(mask, 1):
        m[mod] = i

    return m


def resize_multilabel_mask(mask, factor, axes_to_scale=np.array([0,1,2]), multilabel_mask_splitter=one2many):
    return many2one(np.array([resize_binary_mask(mod.astype(np.float), factor, axes_to_scale)
                       for mod in multilabel_mask_splitter(mask)]))


def split_liver_tumor_mask(mask, n_class=None):
    if n_class is None:
        n_class = int(mask.max())
    res = []
    for i in range(1, n_class + 1):
        if i == 1:
            res.append(mask >= 1)
        else:
            res.append(mask == i)

    return np.stack(res)

class CroppedDataset(SegmentationFromCSV):
    def __init__(self, data_path, liver_pred_path, modalities= ['CT'], target='target', metadata_rpath='meta_with_preds.csv', thresh=0.5):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.df.index = range(len(self.df))
        self.threshold = thresh
        self.pred_path = liver_pred_path
    
    def get_indices(self, i):
        fname = os.path.join(self.pred_path, self.df.loc[i].pred)
        print(fname)
        pred = load(fname)[0][0] > self.threshold
        print(pred.shape)
        ind_min, ind_max = SelectSlice(pred)
        return 2*ind_min, 2*ind_max
        
    def load_image(self, i):
        #fname = os.path.join(self.path, self.df.loc[i].CT)
        img = np.float32(super().load_image(i)[0])  # 4D -> 3D
        ind_min, ind_max = self.get_indices(i)
        return img[:,:,ind_min:ind_max]

    def load_segm(self, i):
        return   # already 3D
        img = np.float32(super().load_segm(i)==2)
        ind_min, ind_max = self.get_indices(i)
        return img[:,:,ind_min:ind_max]

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))
