import dill
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from mdfeature.diffusion_utils import free_energy_estimate_2D
from mdfeature.potentials import ring_double_well_potential


class AreaOfInterest:

    def __init__(self, x_min, x_max, y_min, y_max, x_samples=1000, y_samples=1000):
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
        self.values = None

    def impute_x(self, index):
        return self.x_min + (index/self.x_samples) * self.x_width

    def impute_y(self, index):
        return self.y_min + (index/self.y_samples) * self.y_width

    def indices_to_coordinate(self, x_index, y_index):
        return [self.impute_x(x_index), self.impute_y(y_index)]

    def evaluate_function(self, function):
        values = np.zeros((self.x_samples, self.y_samples))
        for x in range(self.x_samples):
            for y in range(self.y_samples):
                values[x][y] = function(self.indices_to_coordinate(x, y))

        plt.imshow(values)
        plt.show()

        self.values = values

    def detect_local_minima(self):
        assert self.values is not None, "run evaluate_function before detect_local_minima"
        minima = (self.values == scipy.ndimage.minimum_filter(self.values, 3, mode='constant', cval=0.0))
        minima_locations = np.where(1 == minima)
        minima_index_array = np.zeros((len(minima_locations[0]), len(minima_locations)))
        for dim, arr in enumerate(minima_locations):
            for point, val in enumerate(arr):
                minima_index_array[point, dim] = val

        return minima_index_array


class ConvergenceAnalyser:

    def __init__(self, trajectory, reference_potential, area_of_interest: AreaOfInterest, plot_interval=1000):
        self.trajectory = trajectory
        self.plot_interval = plot_interval
        self.AoI = area_of_interest
        self.AoI.evaluate_function(reference_potential)
        local_minima = self.AoI.detect_local_minima()
        print(local_minima)
        free_energy, _, _, x_edges, y_edges = free_energy_estimate_2D(trajectory, beta=1, bins=300)
        print(free_energy.shape)
        print(x_edges.shape)
        print(y_edges.shape)
        print(x_edges)
        print(y_edges)


if __name__ == "__main__":
    traj = dill.load(open("../../notebooks/double_ring_well_traj_phys.pickle", "rb"))
    AoI = AreaOfInterest(x_min=-3, x_max=3, y_min=-3, y_max=3, x_samples=1000, y_samples=1000)
    ca = ConvergenceAnalyser(traj, ring_double_well_potential, AoI)

