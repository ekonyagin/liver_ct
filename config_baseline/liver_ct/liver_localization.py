import numpy as np

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