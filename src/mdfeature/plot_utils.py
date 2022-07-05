import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import warnings

from mdfeature.diffusion_utils import free_energy_estimate, project_points_to_line, free_energy_estimate_2D
from scipy.signal import find_peaks
from math import ceil, floor


def get_digit_text_width(fig, axis):
    r = fig.canvas.get_renderer()
    t = axis.text(0.5, 0.5, "1")

    bb = t.get_window_extent(renderer=r).transformed(axis.transData.inverted())

    t.remove()

    return bb.width / 2


def display_kramers_rates(kramers_rates):
    print("Kramer's Rates")
    print("-"*25)
    for rate in kramers_rates:
        initial_state = rate[0][0]
        final_state = rate[0][1]
        transition_rate = rate[1]
        print(fr"S{initial_state} --> S{final_state} : {'{:e}'.format(transition_rate)}")
    print("-"*25)


def get_minima(data_array, prominence, include_endpoint_minima: bool, ignore_high_minima: bool):
    number_of_minima = 0
    current_prominence = prominence
    minima = None

    while number_of_minima < 2:
        minima = find_peaks(-data_array, prominence=current_prominence)[0]
        energy = [data_array[index] for index in minima]

        if ignore_high_minima:
            minima = [point for idx, point in enumerate(minima) if not np.abs(energy[idx]) > min(data_array)
                      + 0.8*(max(data_array) - min(data_array))]

        number_of_minima = len(minima)
        current_prominence *= 0.975

    if current_prominence != prominence*0.975:
        warnings.warn(f"Automatically reduced prominence from {prominence} "
                      f"to {round(current_prominence / 0.975, 3)} so as to find at least two minima.")

    if include_endpoint_minima:
        if data_array[0] < data_array[1]:
            minima = np.insert(minima, 0, 0)
        if data_array[-1] < data_array[-2]:
            minima = np.append(minima, len(data_array) - 1)

    return minima


def plot_minima(minima_list, y_variable):
    for idx, minima in enumerate(minima_list):
        print("Minima ", (round(minima[0], 3), round(minima[1], 3)))
        plt.text(minima[0], minima[1] - 0.075 * (max(y_variable) - min(y_variable)), f"S{idx}",
                 fontsize=16, color='b')


def display_state_boundaries(msm, y_coordinate):
    voronoi_cell_boundaries = [(msm.sorted_state_centers[i + 1] + msm.sorted_state_centers[i]) / 2
                               for i in range(len(msm.sorted_state_centers) - 1)]
    for boundary in voronoi_cell_boundaries:
        plt.vlines(boundary, ymin=min(y_coordinate) - 0.2 * (max(y_coordinate) - min(y_coordinate)),
                   ymax=min(y_coordinate) - 0.1 * (max(y_coordinate) - min(y_coordinate)), linestyle='--',
                   color='k')

    return voronoi_cell_boundaries


def display_state_numbers(boundaries, x_variable, y_variable, digit_width):
    x_min = min(x_variable)
    y_min = min(y_variable)
    y_range = max(y_variable) - min(y_variable)
    def index_label_width(x): return digit_width * ceil(np.log10(x+1))

    for state_index in range(len(boundaries) + 1):
        if state_index == 0:
            plt.text((boundaries[state_index] + x_min) / 2 - index_label_width(state_index),
                     y_min - 0.175 * y_range, f"{state_index}",
                     fontsize=12, color='k')
        elif state_index == len(boundaries):
            plt.text((x_min + boundaries[state_index - 1]) / 2 - index_label_width(state_index),
                     y_min - 0.175 * y_range, f"{state_index}",
                     fontsize=12, color='k')
        else:
            plt.text((boundaries[state_index] + boundaries[state_index - 1]) / 2
                     - index_label_width(state_index), y_min - 0.175 * y_range, f"{state_index}",
                     fontsize=12, color='k')


def plot_free_energy_landscape(self):
    fig = plt.figure(figsize=(15, 7))
    ax1 = plt.subplot()

    # Diffusion curve
    l1, = ax1.plot(self.smoothed_D_domain, self.smoothed_D, color='red')
    ax1.set_ylim((min(self.smoothed_D)
                  - 0.2*(max(self.smoothed_D) - min(self.smoothed_D)),
                  max(self.smoothed_D)
                  + 0.1*(max(self.smoothed_D) - min(self.smoothed_D))))

    # Free energy curve
    ax2 = ax1.twinx()
    l2, = ax2.plot(self.coordinates, self.smoothed_F, color='k')
    digit_width = get_digit_text_width(fig, ax2)
    plt.legend([l1, l2], ["diffusion_coefficient", "free_energy"])

    print(f"Free energy profile suggests {len(self.free_energy_minima)} minima.")
    plot_minima(self.free_energy_minima, self.smoothed_F)
    state_boundaries = display_state_boundaries(self.msm, self.smoothed_F)
    if len(state_boundaries) < 50:
        display_state_numbers(state_boundaries, self.coordinates, self.smoothed_F, digit_width)

    ax1.set_xlabel('Q', fontsize=16)
    ax1.set_ylabel(r"Diffusion Coefficient ($Q^2 / s$)", fontsize=16)
    ax2.set_ylabel(r"Free Energy ($kJ/mol$)", fontsize=16)
    plt.title('Free Energy Landscape', fontsize=16)
    plt.show()


def plot_free_energy_estimate(potential, samples, beta, minimum_counts=50):
    estimated_free_energy, coordinates = free_energy_estimate(samples, beta, minimum_counts)
    linear_shift = estimated_free_energy[floor(len(estimated_free_energy) / 2)] - potential(0)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(coordinates, estimated_free_energy - linear_shift, 'k', label='estimated')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('F', fontsize=16)
    x_range = np.arange(min(coordinates), max(coordinates), (max(coordinates) - min(coordinates)) / 1000)
    plt.plot(x_range, potential(x_range), label='actual')
    plt.legend()
    plt.title('Free Energy Surface', fontsize=16)


def plot_free_energy_slice(samples, beta, slice_centre, slice_angle, minimum_counts=50):
    concatenated_samples = np.concatenate(samples)
    projected_samples = project_points_to_line(concatenated_samples, np.array(slice_centre), slice_angle)
    free_energy, coordinates = free_energy_estimate(projected_samples, beta, minimum_counts)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(18, 5)
    axs[0].hist2d(concatenated_samples[:, 0], concatenated_samples[:, 1], bins=300)
    axs[0].plot(slice_centre[0], slice_centre[1], 'rx', markersize=12)
    max_x = max(concatenated_samples[:, 0])
    min_x = min(concatenated_samples[:, 0])
    x_range = np.arange(min_x, max_x, (max_x - min_x) / 1000)
    m = np.tan(slice_angle);
    c = slice_centre[1] - m * slice_centre[0]
    y_range = m * x_range + c
    axs[0].plot(x_range, y_range, 'r')
    axs[1].hist(projected_samples, bins=100)
    axs[2].plot(coordinates, free_energy)
    plt.show()

    return projected_samples


def plot_free_energy_surface(samples, beta, bins=300):
    concatenated_samples = np.concatenate(samples)
    free_energy, fig, axs, xedges, yedges = free_energy_estimate_2D(samples, beta, bins=bins)
    fig.set_size_inches(9, 7)
    clb = axs.contourf(xedges[1:], yedges[1:], free_energy.T)
    plt.colorbar(clb, label='Free Energy', ax=axs)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.show()


def hamiltonian(Q, P, M, U):
    return - ((P ** 2) / (2*M) + U(Q))


def phase_plot(Q, P, U, M):
    min_Q = min(Q); max_Q = max(Q); range_Q = max_Q - min_Q;
    min_P = min(P); max_P = max(P); range_P = max_P - min_P;
    Q_range = np.linspace(min_Q - 0.05 * range_Q, max_Q + 0.05 * range_Q, 100)
    P_range = np.linspace(min_P - 0.05 * range_P, max_P + 0.05 * range_P, 100)
    Q_mesh, P_mesh = np.meshgrid(Q_range, P_range)
    plt.pcolormesh(Q_range, P_range, hamiltonian(Q_mesh, P_mesh, M, U))
    plt.contour(Q_range, P_range, hamiltonian(Q_mesh, P_mesh), levels = 15)
    plt.xlabel(r'$Q$', fontsize=20)
    plt.ylabel(r'$P$', fontsize=20)
    plt.plot(Q, P, 'k', linewidth=0.7)
    plt.show()

def trajectory_plot(Q0, Q1, U):
    min_Q0 = min(Q0); max_Q0 = max(Q0); range_Q0 = max_Q0 - min_Q0;
    min_Q1 = min(Q1); max_Q1 = max(Q1); range_Q1 = max_Q1 - min_Q1;
    Q0_range = np.linspace(min_Q0 - 0.05 * range_Q0, max_Q0 + 0.05 * range_Q0, 100)
    Q1_range = np.linspace(min_Q1 - 0.05 * range_Q1, max_Q1 + 0.05 * range_Q1, 100)
    Q0_mesh, Q1_mesh = np.meshgrid(Q0_range, Q1_range)
    plt.pcolormesh(Q0_range, Q1_range, U([Q0_mesh, Q1_mesh]))
    plt.contour(Q0_range, Q1_range, U([Q0_mesh, Q1_mesh]), levels = 15)
    plt.xlabel(r'$Q_0$', fontsize=20)
    plt.ylabel(r'$Q_1$', fontsize=20)
    plt.plot(Q0, Q1, 'k', linewidth=0.7)
    plt.show()

def potential_contour_plot(x_min, x_max, y_min, y_max, U, vmin=None, vmax=None, set_min_to_zero=True, save=True, save_name='test'):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    axs.set_aspect('equal')
    x_range = np.linspace(x_min, x_max, 120)
    y_range = np.linspace(y_min, y_max, 120)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    if set_min_to_zero:
        U_min = np.min(U([x_mesh, y_mesh]))
    else:
        U_min = 0
    im = axs.pcolormesh(x_range, y_range, U([x_mesh, y_mesh])-U_min)
    #axs.contour(x_range, y_range, U([x_mesh, y_mesh]), levels=15)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.20)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label='Free Energy', fontsize=20)
    axs.xaxis.set_ticks([-2, -1, 0, 1, 2])
    axs.yaxis.set_ticks([-2, -1, 0, 1, 2])
    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)
    if vmin is not None and vmax is not None:
        im.set_clim(vmin, vmax)

    if save:
        plt.savefig(save_name, format="pdf", bbox_inches="tight")

    plt.show()