import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pprint

def gaussian_comp_f(l: int, p: np.ndarray) -> np.ndarray:
    """Generate a gaussian 1d array with length L and parameters p

    Args:
        l (int): desired signal length L
        p (np.ndarray): 1x3 np.ndarray of gaussian a, mean and std.

    Returns:
        np.ndarray: gaussian 1d array with length L and parameters p
    """
    x = np.arange(l)
    return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2))


def gaussian_comp_error(p: int, y0: np.ndarray):
    """Compute the difference between the ground truth (y0) and summation of gaussian components

    Args:
        p (int): parameters of gaussian components. reshaped from N x 3 parameter array
        y0 (np.ndarray): ground truth

    Returns:
        error
    """
    p = np.reshape(p, (-1, 3))
    comp_n = p.shape[0]
    signal_length = len(y0)
    fit = np.zeros((signal_length, ))
    for i in range(comp_n):
        fit += gaussian_comp_f(signal_length, p[i])
    return fit - y0

def rrcycle_gaussian_decomp(rrcycle: np.ndarray, p_init: np.ndarray, verbose=False)->dict:
    """Decomposite one PPG RR cycle into guassians 1d numpy arrays

    Args:
        rrcycle (np.ndarray): One segment of PPG signal which contains one RR cycle. Sized L x 1. L refers to the length of the PPG segment.
        p_init (np.ndarray): The initial parameters for least square optimization. Sized N x 3. N refers to the desired number of gaussian components, while every component has 3 initial parameters, i.e. the A, mean, and std.
        verbose (bool, optional): whether prints diagnostic info. Defaults to False.

    Returns:
        dict: decomposition results:
            'gaussian_comps': list of N gaussian components with length L
            'gaussian_parameters': np.ndarray of N x 3

    """
    comp_n = p_init.shape[0]
 
    # generate optimization boundary
    lb = list()
    ub = list()
    for i in range(comp_n):
        for j in range(3):
            if j == 0:
                # a >= 0
                lb.append(0)
                ub.append(np.inf)
            elif j == 1:
                # mean >= 0
                lb.append(0)
                ub.append(np.inf)
            elif j == 2:
                # std. >= 0
                lb.append(0)
                ub.append(np.inf)
     
    fit_result = least_squares(gaussian_comp_error, np.squeeze(np.reshape(p_init, (-1, 1))), args=(rrcycle, comp_n), bounds=(lb, ub))
    p_hat = np.reshape(fit_result.x, (-1, 3))

    comps = list()

    x = np.arange(len(rrcycle))
    rrcycle_fit = np.zeros((len(rrcycle), ))

    for i in range(p_hat.shape[0]):
        comps.append(gaussian_comp_f(len(rrcycle), p_hat[i]))
        rrcycle_fit += comps[-1] 
    
    # sort the component by gaussian mean
    sidx = np.argsort(p_hat[:, 1])
    p_hat = p_hat[sidx]
    
    if verbose:
        pprint.pprint(p_hat)
        fig, axe = plt.subplots(1, 1, figsize=(6, 6), dpi=120)
        axe.plot(rrcycle)
        axe.plot(rrcycle_fit)

        for comp in comps:
            axe.plot(comp)
    
    return {
        'gaussian_comps': comps,
        'gaussian_parameters': p_hat
    }