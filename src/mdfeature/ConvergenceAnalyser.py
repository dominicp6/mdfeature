import warnings
import dill
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from mdfeature.general_utils import select_lowest_minima


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx]


def cart2pol(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return r, theta


def ring_double_well_potential(x):
    theta0 = np.pi
    r0 = 1
    w = 0.2
    d = 5
    r, theta = cart2pol(x[0], x[1])

    return (1 / r) * np.exp(r / r0) - d * np.exp(-((x[0] - r0) ** 2 + (x[1]) ** 2) / (2 * w ** 2)) - d * np.exp(
        -((x[0] - r0 * np.cos(theta0)) ** 2 + (x[1] - r0 * np.sin(theta0)) ** 2) / (2 * w ** 2))


def free_energy_estimate_2D(samples, beta, bins=300):
    hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins)
    total_counts = np.sum(hist)
    with np.errstate(divide='ignore'):
        free_energy = - (1 / beta) * np.log(hist / total_counts + 0.000000000001)
        free_energy = np.nan_to_num(free_energy, nan=0)

    return free_energy - np.min(free_energy), xedges, yedges


class AreaOfInterest:

    def __init__(self, x_min, x_max, y_min, y_max, x_samples=1000, y_samples=1000, values=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_width = x_max - x_min
        self.y_width = y_max - y_min
        self.x_samples = x_samples
        self.y_samples = y_samples
        assert self.x_width > 0, "x_max must be larger than x_min"
        assert self.y_width > 0, "y_max must be larger than y_min"
        if values is not None:
            self.x_samples = values.shape[0]
            self.y_samples = values.shape[1]
            self.values = values
        else:
            self.values = None
        self.minima = None
        self.function = None

    def x_values(self):
        return [self.impute_x(index) for index in range(self.x_samples)]

    def y_values(self):
        return [self.impute_y(index) for index in range(self.y_samples)]

    def grid_points(self):
        return np.array([(self.impute_x(idx_x), self.impute_y(idx_y))
                         for idx_x in range(self.x_samples)
                         for idx_y in range(self.y_samples)])

    def grid_point_values(self):
        return self.values.flatten()

    def impute_x(self, index):
        return self.x_min + (index / self.x_samples) * self.x_width

    def impute_y(self, index):
        return self.y_min + (index / self.y_samples) * self.y_width

    def indices_to_coordinate(self, x_index, y_index):
        return [self.impute_x(x_index), self.impute_y(y_index)]

    def evaluate_function(self, function):
        values = np.zeros((self.x_samples, self.y_samples))
        for x in range(self.x_samples):
            for y in range(self.y_samples):
                values[x][y] = function(self.indices_to_coordinate(x, y))

        print(values[50,150], values[150,50])
        print(function([1,0]), function([0,1]))

        return values

    def grid_interpolation(self, other):
        assert other.values is not None, "other must define values to interpolate over"
        return interpolate.griddata(other.grid_points(),
                                    other.grid_point_values(),
                                    self.grid_points(), method='cubic').reshape((self.x_samples, self.y_samples))

    def detect_local_minima(self, values=None, function=None):
        if values is None and function is not None:
            values = self.evaluate_function(function)
        elif values is not None and function is not None:
            raise ValueError("either a function of values array must be provided")
        else:
            pass
        minima = (values == scipy.ndimage.minimum_filter(values, 50, mode='constant', cval=0.0))
        minima_locations = np.where(1 == minima)
        number_of_minima = len(minima_locations[0])
        number_of_dimensions = len(minima_locations)
        minima_index_array = np.zeros((number_of_minima, number_of_dimensions))
        for dim, arr in enumerate(minima_locations):
            for point, val in enumerate(arr):
                minima_index_array[point, dim] = val

        minima_coordinate_array = [self.indices_to_coordinate(*row_array[0]) for row_array in np.vsplit(minima_index_array, minima_index_array.shape[0])]
        minima = select_lowest_minima(minima_coordinate_array, function, n=2)

        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.x_values(), self.y_values(), values.T)
        im.set_clim(np.min(values), np.percentile(values, 0.999999))
        self.values = values
        self.function = function
        self.minima = minima
        plt.scatter(self.minima[:,0], self.minima[:,1], c='r')
        plt.show()
        return self.minima

    def array_interpolate_function(self):
        """
        Generates a cubic interpolation function for an area of interest.
        """
        cubic_interpolated_array = interpolate.interp2d(self.x_values(), self.y_values(), self.values, kind='cubic')

        return cubic_interpolated_array

    def compute_minima_anomaly(self, other):
        """
        Computes minima anomaly assuming self as the reference.
        """
        assert self.minima is not None, "can only compute minima anomaly after running detect_local_minima"
        assert len(self.minima) == 2, f"2 local minima required, found {len(self.minima)}"
        assert self.function is not None, "to compute minima anomaly, reference instance requires a defining function"
        other_interpolator = other.array_interpolate_function()
        minima1 = self.minima[0]
        minima2 = self.minima[1]
        reference_difference = self.function(minima1) - self.function(minima2)
        other_difference = other_interpolator(*minima1) - other_interpolator(*minima2)
        anomaly = other_difference - reference_difference

        return anomaly

    def compute_rmsd_domain_anomaly(self, other, ignore_nans=True):
        if other.values.shape != self.values.shape:
            interpolated_grid = self.grid_interpolation(other)
            warnings.WarningMessage('Self and other have different grid sizes. '
                                    'Interpolation will result in significant slow downs.')
        else:
            interpolated_grid = other.values
        values = self.values
        if ignore_nans:
            interpolated_grid = np.ma.array(interpolated_grid, mask=np.isnan(interpolated_grid))
            values = np.ma.array(values, mask=np.isnan(values))

        # rel_p_hat = np.exp(- interpolated_grid - np.min(interpolated_grid))
        # p_hat = rel_p_hat / np.sum(rel_p_hat)
        # rel_p = np.exp(-values - np.min(values))
        # p = rel_p / np.sum(rel_p)
        # delta_F = np.sum(p_hat * interpolated_grid) - np.sum(p * values)
        # interpolated_grid -= delta_F

        # plt.imshow(interpolated_grid)
        # plt.show()
        # plt.imshow(values)
        # plt.show()
        # input()

        return np.sqrt(np.mean((interpolated_grid - values) ** 2))


class ConvergenceAnalyser:

    def __init__(self, trajectory, reference_potential, area_of_interest: AreaOfInterest):
        self.trajectory = trajectory
        self.reference_potential = reference_potential
        self.AoI = area_of_interest
        self.local_minima = self.AoI.detect_local_minima(function=reference_potential)

    def plot_anomaly(self, burn_in=100000, plot_interval=2000):
        steps = []
        minima_anomalies = []
        domain_rmsd_anomalies = []
        for step in tqdm(np.arange(plot_interval, len(self.trajectory), plot_interval)):
            free_energy, x_edges, y_edges = free_energy_estimate_2D(self.trajectory[:step], beta=1, bins=300)
            empirical_free_energy = AreaOfInterest(values=free_energy,
                                                   x_min=min(x_edges),
                                                   x_max=max(x_edges),
                                                   y_min=min(y_edges),
                                                   y_max=max(y_edges))
            minima_anomaly = self.AoI.compute_minima_anomaly(empirical_free_energy)
            rmsd_anomaly = self.AoI.compute_rmsd_domain_anomaly(empirical_free_energy, ignore_nans=True)
            if step > burn_in:
                steps.append(step)
                minima_anomalies.append(np.abs(minima_anomaly))
                domain_rmsd_anomalies.append(rmsd_anomaly)

        plt.plot(steps, minima_anomalies, c='k')
        plt.semilogy()
        plt.xlim([0, len(self.trajectory)])
        plt.xlabel('Steps Elapsed')
        plt.ylabel(r'$\vert{\Delta \hat{F}}_{AB} - \Delta F_{AB}\vert$')
        plt.show()

        plt.plot(steps, domain_rmsd_anomalies, c='k')
        plt.xlim([0, len(self.trajectory)])
        plt.xlabel('Steps Elapsed')
        plt.ylabel(r'$\sqrt{\vert \hat{F}_{ij} - F_{ij} \vert_2}$')
        plt.show()


if __name__ == "__main__":
    traj = dill.load(open("../../notebooks/double_ring_well_traj_phys.pickle", "rb"))
    AoI = AreaOfInterest(x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, x_samples=300, y_samples=300)
    ca = ConvergenceAnalyser(traj, ring_double_well_potential, AoI)
    ca.plot_anomaly(plot_interval=5000)
