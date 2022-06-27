import numpy as np
from autograd import grad
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, cpu_count


class LangevinDynamics():

    def __init__(self, x0, potential, beta, time_step):
        # TODO: check not int
        self.x = x0
        try:
            self.dim = len(x0)
        except:
            self.dim = 1
        log_prob = lambda x: - beta * potential(x)

        if self.dim == 1:
            self.gradLogProb = grad(log_prob)
        else:
            self.gradLogProb = lambda x: np.array([float(array) for array in grad(log_prob)(x)])
        self.time_step = time_step

    def simulate(self, number_of_steps, burn_in, seed=0, num_processes=None):
        if num_processes is None:
            num_processes = max(1, cpu_count()-1)
        else:
            assert isinstance(num_processes, int)
        steps_per_process = int(number_of_steps/num_processes)
        assert burn_in < steps_per_process

        with Manager() as manager:
            L = manager.list()  # <-- can be shared between processes.
            processes = []
            for process_id in range(num_processes):
                np.random.seed(process_id + seed*num_processes)
                p = Process(target=self.simulate_single_core, args=(L, steps_per_process, process_id, burn_in))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            return list(L)

    def simulate_single_core(self, trajectory_array, number_of_steps, process_id, burn_in):
        x = self.x
        trajectory = np.zeros((number_of_steps, self.dim))
        if process_id == 0:
            for step in tqdm(range(number_of_steps)):
                x = self.step(x)
                trajectory[step, :] = x
        else:
            for step in range(number_of_steps):
                x = self.step(x)
                trajectory[step, :] = x

        trajectory_array.append(trajectory[burn_in:])

    def step(self, x):
        # Euler-Maruyama Step
        W = np.random.normal(size=self.dim)
        x += self.time_step * self.gradLogProb(x) + np.sqrt(2 * self.time_step) * W

        return x


if __name__ == "__main__":
    def double_well_negative_log(x):
        h = 2
        c = 2
        return -(1 / 4) * (x ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x ** 4)

    x_range = np.arange(-3, 3, 1/1000)
    plt.plot(x_range, [double_well_negative_log(x) for x in x_range])
    plt.show()

    temperature = 300
    R = 0.0083144621  # Universal Gas Constant kJ/K/mol
    beta = 1.0 / (temperature * R)  # units (kJ/mol)**(-1)

    ld = LangevinDynamics(x0=0.0, potential=double_well_negative_log, beta=beta, time_step=5e-3)
    trajectories = ld.simulate(number_of_steps=100000, burn_in=2000)
    combined_trajectory = np.concatenate(trajectories).ravel()
    plt.hist(combined_trajectory, bins=100)
    plt.show()
