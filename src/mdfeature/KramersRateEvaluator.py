import numpy as np
import warnings
import pandas as pd
import pyemma
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from autoimpute.imputations import MiceImputer
from scipy.signal import find_peaks
from itertools import permutations
from math import ceil

def RMS_difference(A, B):
    return np.sqrt(np.mean((A-B)**2))

def interpolate_function(x, y, x_to_evaluate):
    x_min = min(x)
    x_max = max(x)

    if x_min <= x_to_evaluate <= x_max:
        # interpolate
        x_low = max([x for x in x if x <= x_to_evaluate])
        i_low = np.where(x == x_low)
        x_high = min([x for x in x if x > x_to_evaluate])
        i_high = np.where(x == x_high)
        interpolation_distance = (x_to_evaluate - x_low) / (x_high - x_low)

        return float(y[i_low] + (y[i_high] - y[i_low]) * interpolation_distance)

    elif x_to_evaluate < x_min:
        # extrapolate
        return float(y[0] - (x_min - x_to_evaluate) * (y[1] - y[0]) / (x[1] - x[0]))

    elif x_to_evaluate > x_max:
        # extrapolate
        return float(y[-1] + (x_to_evaluate - x_max) * (y[-1] - y[-2]) / (x[-1] - x[-2]))


class MSM():

    def __init__(self, state_centers):
        self.state_centers = state_centers
        self.number_of_states = len(state_centers)
        self.sorted_state_centers = np.sort(state_centers)
        self.stationary_distribution = None
        self.transition_matrix = None

    def compute_diffusion_coefficient_domain(self):
        diffusion_coeff_domain = []
        for idx in range(len(self.sorted_state_centers) - 1):
            diffusion_coeff_domain.append((self.sorted_state_centers[idx+1]+self.sorted_state_centers[idx])/2)

        return diffusion_coeff_domain

    def set_stationary_distribution(self, stationary_distribution):
        self.stationary_distribution = stationary_distribution

    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix


class KramersRateEvaluator():

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.imputer = MiceImputer(strategy={"F": "interpolate"}, n=1, return_list=True)

    def _replace_inf_with_nan(self, array):
        for idx, entry in enumerate(array):
            if entry == np.inf or entry == -np.inf:
                array[idx] = np.nan

        return array

    def _compute_free_energy_surface(self, time_series, beta, bins=200, impute=True):
        counts, coordinates = np.histogram(time_series, bins=bins)
        coordinates = coordinates[:-1]

        if impute:
            with np.errstate(divide='ignore'):
                normalised_counts = counts / np.sum(counts)
                free_energy = (1/beta) * self._replace_inf_with_nan(-np.log(normalised_counts))

            if np.isnan(free_energy).any():
                warnings.warn(f"NaN values were found in the free energy calculation. "
                              f"Consider using a longer trajectory or rerunning with fewer bins (currently bins={bins}). "
                              f"Fixing with imputation for now.")
                if self.verbose is True:
                    print(f'Note: Of the {len(free_energy)} free energy evaluations, '
                          f'{np.count_nonzero(np.isnan(free_energy))} were NaN values.')
                df = pd.DataFrame({'CV': coordinates, 'F': free_energy})
                output_df = self.imputer.fit_transform(df)[0][1]
                free_energy = output_df.F
            else:
                pass
        else:
            robust_counts = counts[np.where(counts > 25)]
            normalised_counts = robust_counts / np.sum(counts)
            free_energy = - (1/beta) * np.log(normalised_counts)
            coordinates = coordinates[np.where(counts > 25)]


        self.coordinates = coordinates
        self.free_energy = free_energy

    def _compute_coordinate_mean_sq_difference(self):
        self.msd_coordinate = np.sqrt(np.mean([(self.coordinates[i]-self.coordinates[i+1])**2
                                       for i in range(len(self.coordinates)-1)]))

    def _relabel_trajectory_by_coordinate_chronology(self, traj):
        sorted_indices = np.argsort(np.argsort(self.msm.state_centers))

        # relabel states in trajectory
        for idx, state in enumerate(traj):
            traj[idx] = sorted_indices[traj[idx]]

        return traj

    def _compute_counts_matrix(self, traj, lag):
        num_of_states = self.msm.number_of_states
        counts = np.zeros((num_of_states, num_of_states))
        for idx, state in enumerate(traj[:-lag]):
            counts[state, traj[idx+lag]] += 1

        return counts

    def _compute_optimal_lag_time(self, step_size, discrete_traj, sigmaD, sigmaF, lag_times):
        old_diffusion_coefficients = None
        diffs = []
        for lag in lag_times:
            self.msm = pyemma.msm.estimate_markov_model(dtrajs=discrete_traj, lag=lag)
            self._compute_diffusion_coefficients(step_size, lag)
            self._smooth_D_and_F(sigmaD=sigmaD, sigmaF=sigmaF)
            if old_diffusion_coefficients is not None:
                diff = RMS_difference(old_diffusion_coefficients, self.smoothed_diffusion_coefficients)
                diffs.append(diff)
            old_diffusion_coefficients = self.diffusion_coefficients

        index_optimal_lag = np.argmin(diffs)
        optimal_lag = lag_times[index_optimal_lag]

        if self.verbose:
            plt.plot(lag_times[:-1], diffs, c='k')
            plt.xlabel('Lag', fontsize=16)
            plt.ylabel(r'RMSD $D(Q)$', fontsize=16)
            plt.vlines(index_optimal_lag, min(diffs), max(diffs), c='r')
            plt.show()

        return optimal_lag

    def _fit_MSM(self, time_series,
                 cluster_type='kmeans',
                 options={'k': 10, 'stride': 5, 'max_iter': 150,
                          'max_centers': 100, 'metric':'euclidean', 'n_jobs': None, 'dmin': None},
                 lag=1):

        if cluster_type == 'kmeans':
            cluster = pyemma.coordinates.cluster_kmeans(time_series,
                                                        k=options['k'],
                                                        stride=options['stride'],
                                                        max_iter=options['max_iter'])
        elif cluster_type == 'reg_space':
            if options['dmin'] is None:
                options['dmin'] = min(10 * self.msd_coordinate, (max(self.coordinates)-min(self.coordinates))/10)
                warnings.warn(f"reg_space requires dmin value which was not set in options, "
                              f"setting it automatically to {round(options['dmin'],4)}.")
            cluster = pyemma.coordinates.cluster_regspace(time_series, dmin=options['dmin'], max_centers=options['max_centers'])
        else:
            raise ValueError('cluster_type must be either "kmeans" or "reg_space"')

        discrete_traj = cluster.dtrajs[0]
        cluster_centers = cluster.clustercenters.flatten()
        self.msm = MSM(state_centers=cluster_centers)
        discrete_traj = self._relabel_trajectory_by_coordinate_chronology(traj=discrete_traj)

        # Implied Timescale Analysis
        if self.verbose:
            print('MSM Implied Timescale Analysis')
            lags = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
            its = pyemma.msm.its(discrete_traj, lags=lags, nits=10, reversible=True, connected=True)
            pyemma.plots.plot_implied_timescales(its, ylog=False)
            plt.show()

        msm = pyemma.msm.estimate_markov_model(dtrajs=discrete_traj, lag=lag)
        self.msm.set_stationary_distribution(msm.stationary_distribution)
        self.msm.set_transition_matrix(msm.transition_matrix)

        # Chapman-Kolmogorov Test
        #if self.verbose:
        #print('MSM Chapman-Kolmogorov Test')
        #ck_test = msm.cktest(min(msm.nstates, 4))
        #pyemma.plots.plot_cktest(ck_test)
        #plt.show()

        print(f'MSM created with {self.msm.number_of_states} states, using lag time {lag}.')
        fig = plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.msm.transition_matrix)
        plt.xlabel('j', fontsize=16)
        plt.ylabel('i', fontsize=16)
        plt.title(r'MSM Transition Matrix $\mathbf{P}$', fontsize=16)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.plot(self.msm.stationary_distribution, color='k')
        plt.xlabel('i', fontsize=16)
        plt.ylabel(r'$\pi(i)$', fontsize=16)
        plt.title(r'MSM Stationary Distribution $\mathbf{\pi}$', fontsize=16)
        plt.show()


    def _compute_diffusion_coefficients(self, step_size, lag):
        tau = lag * step_size

        def calculate_c(X, n, P):
            return np.sum([(X[j] - X) ** n * P[:, j] for j in range(len(X))], axis=0)

        c1 = calculate_c(self.msm.sorted_state_centers, n=1, P=self.msm.transition_matrix)
        c2 = calculate_c(self.msm.sorted_state_centers, n=2, P=self.msm.transition_matrix)

        self.diffusion_coefficients = (c2 - c1**2)/(2*tau)

        if self.verbose:
            # Perform Langevin dynamics check
            print('Langevin Dynamics Test')
            c4 = calculate_c(self.msm.sorted_state_centers, n=4, P=self.msm.transition_matrix)
            D4 = (1 / (4 * 3 * 2 * tau)) * c4
            plt.plot(self.msm.sorted_state_centers, D4/self.diffusion_coefficients**2, c='k')
            plt.title("Langevin Dynamics Check", fontsize=16)
            plt.ylabel(r"$D^{(4)}(Q)/D^{(2)}(Q)^2$", fontsize=16)
            plt.xlabel(r"$Q$", fontsize=16)
            plt.show()

    def _gaussian_smooth(self, x, y, dx, sigma, gaussian_width=3):
        interp = interpolate.interp1d(x, y, fill_value='extrapolate')
        interpolated_x = np.arange(min(x), max(x)+dx/2, dx)
        interpolated_y = interp(interpolated_x)
        gaussian_x = np.arange(- gaussian_width * sigma, gaussian_width * sigma, dx)
        # multiply by dx to ensure area conservation after convolution
        gaussian = dx * (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(gaussian_x / sigma) ** 2 / 2)
        smoothed_y = np.convolve(interpolated_y, gaussian, mode='same')

        return interpolated_x, smoothed_y

    def _smooth_D_and_F(self, sigmaD=0.01, sigmaF=0.01):
        self.smoothed_diff_coefficients_range, self.smoothed_diffusion_coefficients \
            = self._gaussian_smooth(x=self.msm.sorted_state_centers, y=self.diffusion_coefficients,
                                    dx=self.msd_coordinate, sigma=sigmaD)
        self.smoothed_free_energy_range, self.smoothed_free_energy \
            = self._gaussian_smooth(x=self.coordinates, y=self.free_energy, dx=self.msd_coordinate, sigma=sigmaF)

    def _get_digit_text_width(self, fig, axis):
        r = fig.canvas.get_renderer()
        t = axis.text(0.5, 0.5, "1")

        bb = t.get_window_extent(renderer=r).transformed(axis.transData.inverted())

        t.remove()

        return bb.width/2

    def _plot_free_energy_landscape(self):
        fig = plt.figure(figsize=(15, 7))
        ax1 = plt.subplot()
        l1, = ax1.plot(self.smoothed_diff_coefficients_range, self.smoothed_diffusion_coefficients, color='red')
        ax1.set_ylim((min(self.smoothed_diffusion_coefficients)
                      -0.2*(max(self.smoothed_diffusion_coefficients)-min(self.smoothed_diffusion_coefficients)),
                      max(self.smoothed_diffusion_coefficients)
                      +0.1*(max(self.smoothed_diffusion_coefficients)-min(self.smoothed_diffusion_coefficients))))
        ax2 = ax1.twinx()
        l2, = ax2.plot(self.coordinates, self.smoothed_free_energy, color='k')
        digit_width = self._get_digit_text_width(fig, ax2)
        plt.legend([l1, l2], ["diffusion_coefficient", "free_energy"])
        if self.verbose:
            print(f"Free energy profile suggests {len(self.free_energy_minima)} minima.")
        for idx, minima in enumerate(self.free_energy_minima):
            print("Minima ", (round(minima[0], 3), round(minima[1], 3)))
            plt.text(minima[0], minima[1]-0.075*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy)), f"S{idx}",
                     fontsize=16, color='b')
        voronoi_cell_boundaries = [(self.msm.sorted_state_centers[i+1]+self.msm.sorted_state_centers[i])/2
                                   for i in range(len(self.msm.sorted_state_centers)-1)]
        for boundary in voronoi_cell_boundaries:
            plt.vlines(boundary, ymin=min(self.smoothed_free_energy)-0.2*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy)), ymax=min(self.smoothed_free_energy)-0.1*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy)), linestyle='--', color='k')
        # only display state numbers when there are a relatively small number of states
        if len(voronoi_cell_boundaries) < 50:
            for state_index in range(len(voronoi_cell_boundaries)+1):
                if state_index == 0:
                    plt.text((voronoi_cell_boundaries[state_index]+min(self.coordinates))/2 - digit_width*ceil(np.log10(state_index+1)),
                             min(self.smoothed_free_energy)-0.175*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy)), f"{state_index}",
                             fontsize=12, color='k')
                elif state_index == len(voronoi_cell_boundaries):
                    plt.text((max(self.coordinates)+voronoi_cell_boundaries[state_index-1])/2 - digit_width*ceil(np.log10(state_index+1)),
                             min(self.smoothed_free_energy)-0.175*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy)), f"{state_index}",
                             fontsize=12, color='k')
                else:
                    plt.text((voronoi_cell_boundaries[state_index]+voronoi_cell_boundaries[state_index-1])/2 - digit_width*ceil(np.log10(state_index+1)),
                             min(self.smoothed_free_energy)-0.175*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy)), f"{state_index}",
                             fontsize=12, color='k')
        ax1.set_xlabel('Q', fontsize=16)
        ax1.set_ylabel(r"Diffusion Coefficient ($Q^2 / s$)", fontsize=16)
        ax2.set_ylabel(r"Free Energy ($kJ/mol$)", fontsize=16)
        plt.title('Free Energy Landscape', fontsize=16)
        plt.show()

    def _compute_well_integrand(self, beta, free_energy):
        return [np.exp(- beta * free_energy[x]) for x in range(len(free_energy))]

    def _compute_barrier_integrand(self, beta, free_energy):
        return [np.exp(beta * free_energy[x])
                /interpolate_function(self.smoothed_diff_coefficients_range,
                                      self.smoothed_diffusion_coefficients,
                                      self.coordinates[x]) for x in range(len(free_energy))]

    def _compute_well_and_barrier_integrals(self, initial_x, final_x, mid_x, well_integrand, barrier_integrand):
        if final_x > initial_x:
            well_integral = integrate.simpson(well_integrand[initial_x: mid_x + 1], self.coordinates[initial_x: mid_x + 1])
            barrier_integral = integrate.simpson(barrier_integrand[initial_x + 1: final_x],
                                                 self.coordinates[initial_x + 1:final_x])
        else:
            well_integral = integrate.simpson(well_integrand[mid_x: initial_x+1], self.coordinates[mid_x: initial_x+1])
            barrier_integral = integrate.simpson(barrier_integrand[final_x + 1: initial_x],
                                                 self.coordinates[final_x + 1: initial_x])

        return well_integral, barrier_integral

    def _print_kramers_rates(self, kramers_rates):
        print("Kramer's Rates")
        print("-"*25)
        for rate in kramers_rates:
            initial_state = rate[0][0]
            final_state = rate[0][1]
            transition_rate = rate[1]
            print(fr"S{initial_state} --> S{final_state} : {'{:e}'.format(transition_rate)}")
        print("-"*25)

    def _get_minima(self, minima_prominance, include_endpoint_minima, ignore_high_energy_minima):
        number_of_minima = 0
        prominance = minima_prominance
        free_energy_minima = None
        while number_of_minima < 2:
            free_energy_minima = find_peaks(-self.smoothed_free_energy,
                                            prominence=prominance)[0]
            energy = [self.smoothed_free_energy[index] for index in free_energy_minima]
            if ignore_high_energy_minima:
                free_energy_minima = [minima for idx, minima in enumerate(free_energy_minima)
                                      if not np.abs(energy[idx]) > min(self.smoothed_free_energy)
                                      + 0.8*(max(self.smoothed_free_energy)-min(self.smoothed_free_energy))]
            number_of_minima = len(free_energy_minima)
            prominance *= 0.975

        if prominance != minima_prominance*0.975:
            warnings.warn(f"Automatically reduced prominance from {minima_prominance} to {round(prominance/0.975,3)} so as to find at least two minima.")

        if include_endpoint_minima:
            if self.smoothed_free_energy[0] < self.smoothed_free_energy[1]:
                free_energy_minima = np.insert(free_energy_minima, 0, 0)
            if self.smoothed_free_energy[-1] < self.smoothed_free_energy[-2]:
                free_energy_minima = np.append(free_energy_minima, len(self.smoothed_free_energy)-1)

        return free_energy_minima

    def _compute_kramers_rates(self, beta, sigmaD, sigmaF, minima_prominance, include_endpoint_minima, ignore_high_energy_minima):
        self._smooth_D_and_F(sigmaD, sigmaF)
        free_energy_minima = self._get_minima(minima_prominance, include_endpoint_minima, ignore_high_energy_minima)
        self.free_energy_minima = [(self.coordinates[minima], self.smoothed_free_energy[minima])
                                   for minima in free_energy_minima]
        self._plot_free_energy_landscape()

        well_integrand = self._compute_well_integrand(beta, self.smoothed_free_energy)
        barrier_integrand = self._compute_barrier_integrand(beta, self.smoothed_free_energy)

        kramers_rates = []
        possible_transitions = permutations(range(len(free_energy_minima)), 2)
        for transition in possible_transitions:
            initial_x = free_energy_minima[transition[0]]
            final_x = free_energy_minima[transition[1]]
            mid_x = int(np.floor((initial_x+final_x)/2))
            well_integral, barrier_integral = self._compute_well_and_barrier_integrals(initial_x,
                                                                                       final_x,
                                                                                       mid_x,
                                                                                       well_integrand,
                                                                                       barrier_integrand)
            kramers_rate = (barrier_integral * well_integral)**(-1)
            kramers_rates.append((transition, kramers_rate))

        if self.verbose is True:
            self._print_kramers_rates(kramers_rates)

        return kramers_rates

    def fit(self,
            CV_time_series,
            beta,
            sigmaD,
            sigmaF,
            step_size,
            impute_free_energy_nans=True,
            ignore_high_energy_minima=True,
            cluster_type='kmeans',
            options={'k': 10, 'stride': 5, 'max_iter': 150,
                     'max_centers': 1000, 'metric': 'euclidean',
                     'n_jobs': None, 'dmin': None},
            lag=1,
            bins=200,
            minima_prominance=1.5,
            include_endpoint_minima=True):

        self._compute_free_energy_surface(time_series=CV_time_series, beta=beta, bins=bins, impute=impute_free_energy_nans)
        self._compute_coordinate_mean_sq_difference()
        self._fit_MSM(time_series=CV_time_series,
                      lag=lag,
                      cluster_type=cluster_type,
                      options=options)
        self._compute_diffusion_coefficients(step_size, lag)
        kramers_rates = self._compute_kramers_rates(beta, sigmaD, sigmaF, minima_prominance, include_endpoint_minima, ignore_high_energy_minima)

        return kramers_rates
