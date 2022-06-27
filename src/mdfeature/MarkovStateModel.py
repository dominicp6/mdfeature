import numpy as np
import matplotlib.pyplot as plt


class MSM:

    def __init__(self, state_centers):
        self.state_centers = state_centers
        self.number_of_states = len(state_centers)
        self.sorted_state_centers = np.sort(state_centers)
        self.stationary_distribution = None
        self.transition_matrix = None
        self.lag = None

    def compute_diffusion_coefficient_domain(self):
        diffusion_coeff_domain = []
        for idx in range(len(self.sorted_state_centers) - 1):
            diffusion_coeff_domain.append((self.sorted_state_centers[idx+1]+self.sorted_state_centers[idx])/2)

        return diffusion_coeff_domain

    def set_stationary_distribution(self, stationary_distribution):
        self.stationary_distribution = stationary_distribution

    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def set_lag(self, lag):
        self.lag = lag

    def calculate_correlation_coefficient(self, n):
        assert self.transition_matrix is not None
        return np.sum([(self.sorted_state_centers[j] - self.sorted_state_centers) ** n * self.transition_matrix[:, j]
                       for j in range(self.number_of_states)], axis=0)

    def relabel_trajectory_by_coordinate_chronology(self, traj):
        sorted_indices = np.argsort(np.argsort(self.state_centers))

        # relabel states in trajectory
        for idx, state in enumerate(traj):
            traj[idx] = sorted_indices[traj[idx]]

        return traj

    def compute_diffusion_coefficient(self, time_step, lag):
        tau = lag * time_step
        c1 = self.calculate_correlation_coefficient(n=1)
        c2 = self.calculate_correlation_coefficient(n=2)
        # space-dependent diffusion coefficient
        diffusion_coefficient = (c2 - c1 ** 2) / (2 * tau)

        return diffusion_coefficient

    def plot(self):
        print(f'MSM created with {self.number_of_states} states, using lag time {self.lag}.')
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.transition_matrix)
        plt.xlabel('j', fontsize=16)
        plt.ylabel('i', fontsize=16)
        plt.title(r'MSM Transition Matrix $\mathbf{P}$', fontsize=16)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.plot(self.stationary_distribution, color='k')
        plt.xlabel('i', fontsize=16)
        plt.ylabel(r'$\pi(i)$', fontsize=16)
        plt.title(r'MSM Stationary Distribution $\mathbf{\pi}$', fontsize=16)
        plt.show()
