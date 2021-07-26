import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
import numpy as np

def gaus(x, a, x0, sigma):
    return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_fit(y):
    n = len(y)  # the number of data
    x = ar(range(n))

    mean = n/2
    sigma = 20
    a = np.max(y)

    popt, pcov = curve_fit(gaus, x, y, p0=[a, mean, sigma])

    # plt.plot(x, y, 'b+:', label='data')
    # plt.plot(x, gaus(x, *popt), 'ro:', label='fit')
    # plt.legend()
    # plt.show()
    return popt, pcov



def kspacelines_gauss_fit(kspace_lines):
    dims = kspace_lines.shape
    kspace_lines = np.reshape(kspace_lines, [dims[0], dims[1]*dims[2], dims[3]])
    step = dims[1]*dims[2] // 100
    popt_avg = [0, 0, 0]
    n = 0
    for i in range(0,kspace_lines.shape[1], step):
        try:
            popt, pcov = gauss_fit(np.sqrt(kspace_lines[:, i, 0] ** 2 + kspace_lines[:, i, 1] ** 2))
            popt_avg += popt
            n += 1
        except:
            pass
    return popt_avg / n


def kspaceImg_gauss_fit(kspace):
    dims = kspace.shape
    kspace = np.fft.fftshift(kspace, axes=(0,1))
    cl_idx = (dims[0]//2-1, dims[0]//2, dims[0]//2+1)
    popt_avg = [0, 0, 0]
    n = 0
    for ch in range(0, dims[2]):
        for j in cl_idx:
            try:
                popt1, pcov = gauss_fit(np.sqrt(kspace[:, j, ch, 0] ** 2 + kspace[:, j, ch, 1] ** 2))
                popt2, pcov = gauss_fit(np.sqrt(kspace[j, :, ch, 0] ** 2 + kspace[j, :, ch, 1] ** 2))
                popt = popt1 + popt2
                popt_avg += popt
                n += 2
            except:
                pass
    return popt_avg / n


def kspaceImg_gauss_fit2(kspace):
    dims = kspace.shape
    # kspace = np.fft.fftshift(kspace, axes=(0,1))
    cl_idx = (dims[2]//2-1, dims[2]//2, dims[2]//2+1)
    popt_avg = [0, 0, 0]
    n = 0
    popt_sl = np.zeros((dims[0], 3))
    for sl in range(0, dims[0]):
        for ch in range(0, dims[1]):
            for j in cl_idx:
                try:
                    popt1, pcov = gauss_fit(np.sqrt(kspace[sl, ch, :, j, 0] ** 2 + kspace[sl, ch, :, j, 1] ** 2))
                    popt2, pcov = gauss_fit(np.sqrt(kspace[sl, ch, j, :, 0] ** 2 + kspace[sl, ch, j, :, 1] ** 2))
                    popt = popt1 + popt2
                    popt_avg += popt
                    n += 2
                except:
                    pass
        popt_sl[sl,:] = popt_avg / n

    return popt_sl
