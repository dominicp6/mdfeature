import numpy as np
import scipy.interpolate as interpolate


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


def gaussian_smooth(x, y, dx, sigma, gaussian_width=3):
    # TODO: unit test Gaussian smoothing
    interp = interpolate.interp1d(x, y, fill_value='extrapolate')
    interpolated_x = np.arange(min(x), max(x)+dx/2, dx)
    interpolated_y = interp(interpolated_x)
    gaussian_x = np.arange(- gaussian_width * sigma, gaussian_width * sigma, dx)
    # multiply by dx to ensure area conservation after convolution
    gaussian = dx * (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(gaussian_x / sigma) ** 2 / 2)
    smoothed_y = np.convolve(interpolated_y, gaussian, mode='same')

    return interpolated_x, smoothed_y
