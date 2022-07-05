import numpy as np
from autograd import grad
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, cpu_count


class LangevinDynamics():

    def __init__(self, x0, potential, beta, time_step, D=1):
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
        self.D = D

    def simulate(self, number_of_steps, burn_in, seed=0, num_processes=None):
        if num_processes is None:
            num_processes = max(1, cpu_count() - 1)
        else:
            assert isinstance(num_processes, int)
        steps_per_process = int(number_of_steps / num_processes)
        assert burn_in < steps_per_process

        with Manager() as manager:
            L = manager.list()  # <-- can be shared between processes.
            processes = []
            for process_id in range(num_processes):
                np.random.seed(process_id + seed * num_processes)
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
        x += self.time_step * self.gradLogProb(x) + np.sqrt(2 * self.D * self.time_step) * W

        return x


class PhysicalLangevinDynamics:

    def __init__(self, U, gamma, M, T, Q0, time_step):
        # MX'' = - grad U - gamma*M*x' + sqrt(2*M*gamma*k_b*T) R(t)
        self.U = U
        self.gamma = gamma
        self.M = M
        self.T = T
        self.Q = Q0
        self.time_step = time_step

        try:
            self.dim = len(Q0)
            self.grad_U = lambda x: np.array([float(array) for array in grad(U)(x)])
            self.P = np.zeros(shape=(len(Q0),))
        except:
            self.dim = 1
            self.grad_U = grad(U)
            self.P = 0

    def simulate(self, number_of_steps, burn_in=0):
        trajectory = np.zeros((number_of_steps, self.dim))
        P_trajectory = np.zeros((number_of_steps, self.dim))
        for step_num in tqdm(range(number_of_steps)):
            Q, P = self.step()
            trajectory[step_num, :] = Q
            P_trajectory[step_num, :] = P

        return trajectory[burn_in:], P_trajectory[burn_in:]

    def step(self):
        W = np.random.normal(size=self.dim)
        self.Q += self.P * self.time_step
        self.P += (-self.grad_U(self.Q) - self.gamma * self.P) * self.time_step \
                  + np.sqrt(2 * self.M * self.gamma * self.T * self.time_step) * W

        return self.Q, self.P


class MarkovProcess:

    def __init__(self, drift, diffusion, jump_frequency, jump_amplitude, time_step):
        self.x = 0.0
        self.drift = grad(drift)
        self.diffusion = diffusion
        self.jump_frequency = jump_frequency
        self.jump_amplitude = jump_amplitude
        self.time_step = time_step

    def step(self):
        self.x += self.drift(self.x)*self.time_step + np.random.normal(scale=self.diffusion)*np.sqrt(self.time_step)
        w = np.random.uniform(0,1)
        if w < (1/self.jump_frequency):
            self.x += np.random.normal(scale=self.jump_amplitude)

        return self.x

    def simulate(self, number_of_steps):
        traj = []
        for step_num in range(number_of_steps):
            traj.append(self.step())

        return traj


if __name__ == "__main__":
    def double_well_negative_log(x):
        h = 2
        c = 2
        return -(1 / 4) * (x ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x ** 4)


    x_range = np.arange(-3, 3, 1 / 1000)
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



