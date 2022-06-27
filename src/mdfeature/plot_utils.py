import matplotlib.pyplot as plt
import numpy as np
import warnings

from scipy.signal import find_peaks
from math import ceil


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
