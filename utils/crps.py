import numpy as np

def crps(true, pred, sample_weight=None, norm=False):
    """Calculate Continuous Ranked Probability Score
    Data based on size (N, ...) where N=number of samples
    Args:
     * true : np.array (...) 
     * pred : np.array (n, ...) 
    """
    num_samples = pred.shape[0]
    print(num_samples)
    diff = pred[1:] - pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, tuple(range(1, pred.ndim)))
    weight = np.broadcast_to(weight,  pred[:-1].shape)
    absolute_error = np.mean(np.abs(pred - np.expand_dims(true, 0)), axis=(0)) 
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    if norm:
        crps_normalized = np.where(np.abs(y_true)> 1E-14, per_obs_crps/np.abs(y_true), np.nan)
        return np.nanmean(crps_normalized, axis=0)
    return np.average(per_obs_crps, axis=0, weights=sample_weight)