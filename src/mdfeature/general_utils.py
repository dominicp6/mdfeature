import numpy as np
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter1d


def replace_inf_with_nan(array):
    for idx, entry in enumerate(array):
        if entry == np.inf or entry == -np.inf:
            array[idx] = np.nan

    return array


def rms_interval(array):
    return np.sqrt(np.mean([(array[i]-array[i+1])**2 for i in range(len(array)-1)]))


def vector_rmsd(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def linear_interp_coordinate_data(x_data: list[float], y_data: list[float], x_to_evaluate: float) -> float:
    x_min = min(x_data)
    x_max = max(x_data)

    if x_min <= x_to_evaluate <= x_max:
        # interpolate
        x_low = max([x for x in x_data if x <= x_to_evaluate])
        i_low = np.where(x_data == x_low)
        x_high = min([x for x in x_data if x > x_to_evaluate])
        i_high = np.where(x_data == x_high)
        interpolation_distance = (x_to_evaluate - x_low) / (x_high - x_low)

        return float(y_data[i_low] + (y_data[i_high] - y_data[i_low]) * interpolation_distance)

    elif x_to_evaluate < x_min:
        # extrapolate
        return float(y_data[0] - (x_min - x_to_evaluate) * (y_data[1] - y_data[0]) / (x_data[1] - x_data[0]))

    elif x_to_evaluate > x_max:
        # extrapolate
        return float(y_data[-1] + (x_to_evaluate - x_max) * (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2]))


def gaussian_smooth(x, y, dx, sigma):
    interp = interpolate.interp1d(x, y, fill_value='extrapolate')
    interpolated_x = np.arange(min(x), max(x)+dx/2, dx)
    sigma_gaussian = sigma/dx
    smoothed_y = gaussian_filter1d(interp(interpolated_x), sigma_gaussian, mode='nearest')

    return interpolated_x, smoothed_y


def select_lowest_minima(minima_array, function, n=2):
    value_array = []
    for minima in minima_array:
        value_array.append(function(minima))
    idx = np.argsort(value_array)[:n]

    return np.array([minima_array[i] for i in idx])

