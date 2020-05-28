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
    max_z = np.minimum(max_z+epsilon, len(sum_distr))

    sum_x = pred.sum(axis=2).sum(axis=0)
    min_x = np.argwhere(sum_x>100).min()
    max_x = np.argwhere(sum_x>100).max()       
    min_x = np.maximum(min_x-epsilon,0)
    max_x = np.minimum(max_x+epsilon, len(sum_x))

    sum_y = pred.sum(axis=2).sum(axis=1)
    min_y = np.argwhere(sum_y>100).min()
    max_y = np.argwhere(sum_y>100).max()
    min_y = np.maximum(min_y-epsilon,0)
    max_y = np.minimum(max_y+epsilon, len(sum_y))

    pred = pred[min_y:max_y, min_x: max_x, min_z:max_z]
    #print(min_y, max_y, sum_y.shape)
    #pred.shape
    return pred