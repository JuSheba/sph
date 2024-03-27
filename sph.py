import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool


def measure_time(func):
    """
    Decorator function to measure the execution time of another function.

    Example:
    --------
    @measure_time
    def my_function(x, y):
        # some time-consuming computation here
        return x + y

    result = my_function(3, 4)
    # Output: 
    # Execution time for my_function  :  0.000123 seconds
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time for", func.__name__, " : ", execution_time, "seconds")
        return result

    return wrapper


# ---------------------------------------------------------------------------------------------------
#        THE MODEL PARAMETERS FROM JSON
# ---------------------------------------------------------------------------------------------------

def init_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='param_conf.json', help='Configuration path')

    args = parser.parse_args()
    global conf
    if args.conf:
        with open(args.conf, 'r', encoding="utf-8") as f:
            conf = json.load(f)


conf = {}
init_conf()


# ---------------------------------------------------------------------------------------------------
#        PARTICLE INITIALIZATION
# ---------------------------------------------------------------------------------------------------

def get_init_coords_with_random_shift(num_particles, min_disk_radius, max_disk_radius, max_shift):
    """
    Generate initial particle coordinates as a two-dimensional array. 
    The particles are generated within 2 disk radii around the origin (0, 0) 
    with random shifts applied.

    Parameters:
    ----------
    num_particles : int
        Number of particles.
    min_disk_radius : float
        Minimum radius of the disk.
    max_disk_radius : float
        Maximum radius of the disk.
    max_shift : float
        Maximum shift allowed from the circular trajectory.

    Returns:
    -------
    numpy.ndarray
        Array containing initial particle coordinates with random shift. 
        The array has a shape of (num_particles, 2), where each row 
        represents the coordinates (x, y) of a particle.
    """
    init_particle_coords = np.zeros((num_particles, 2))
    angles = np.linspace(0, 2 * np.pi, num_particles)
    radii = np.random.uniform(min_disk_radius, max_disk_radius, num_particles)

    init_particle_coords[:, 0] = radii * np.cos(angles) + np.random.uniform(-max_shift, max_shift, num_particles)
    init_particle_coords[:, 1] = radii * np.sin(angles) + np.random.uniform(-max_shift, max_shift, num_particles)
    return init_particle_coords


def get_init_velocities(particle_coords, star_mass):
    """
    Generate initial velocities for particles in a gravitational system 
    with a massive star. The initial velocities are determined based on 
    the orbital velocities of particles around the central star.

    Parameters:
    ----------
    particle_coords : numpy.ndarray
        Array containing initial particle coordinates with shape (N, 2), 
        where N is the number of particles and each row represents 
        the coordinates (x, y) of a particle.
    star_mass : float
        Mass of the central star.

    Returns:
    -------
    numpy.ndarray
        Array containing initial particle velocities with shape (N, 2), 
        where each row represents the velocity components (vx, vy) of a particle.
    """
    def get_orbital_velocity(particle_coords, star_mass):
        distances = np.linalg.norm(particle_coords, axis=1)
        v_orb = np.sqrt(G * star_mass / distances)
        return v_orb
    
    G = 6.67430e-11
    init_particle_velocities = np.zeros_like(particle_coords, dtype=float)
    v_orb = get_orbital_velocity(particle_coords, star_mass)
    theta = np.arctan2(particle_coords[:, 1], particle_coords[:, 0])
    init_particle_velocities[:, 0] = - v_orb * np.sin(theta)
    init_particle_velocities[:, 1] = v_orb * np.cos(theta)
    return init_particle_velocities


def get_masses(num_particles):
    """
    Generate equal masses for all particles.

    Returns:
    -------
    numpy.ndarray
        Array containing masses of particles with shape (N, 1), 
        where N is the number of particles and each row represents the mass of a particle.
    """
    mass_array = np.ones((num_particles, 1)) / num_particles
    return mass_array


def get_init_smoothed_param(num_particles):
    """
    Generate initial smoothing lengths h for all particles.

    Returns:
    -------
    numpy.ndarray
        Array containing smoothing lengths of particles with shape (N, 1), 
        where N is the number of particles and each row represents the smoothing length of a particle.
    """
    init_smoothed_param = np.full((num_particles, 1), 0.5)
    return init_smoothed_param.T[0]


def get_init_energies(num_particles):
    """
    Generate initial energies for all particles.

    Returns:
    -------
    numpy.ndarray
        Array containing initial energies of particles with shape (N, 1), 
        where N is the number of particles and each row represents the energy of a particle.
    """
    energies = np.random.uniform(0.999, 1.0, num_particles)
    return energies


def get_init_pressures(num_particles):
    """
    Generate initial pressures for all particles.

    Returns:
    -------
    numpy.ndarray
        Array containing initial pressures of particles with shape (N, 1), 
        where N is the number of particles and each row represents the pressure of a particle.
    """
    pressures = np.full((num_particles, 1), 0.1)
    return pressures.T[0]


def get_init_densities(num_particles):
    """
    Generate initial densities for all particles.

    Returns:
    -------
    numpy.ndarray
        Array containing initial densities of particles with shape (N, 1), 
        where N is the number of particles and each row represents the density of a particle.
    """
    densities = np.random.uniform(0.999, 0.1, num_particles)
    return densities


def get_time_array(time_point, steps):
    """
    Generate time array for calculation.

    Parameters:
    ----------
    time_point : float
        The end time point for calculation.
    steps : int
        The number of steps in [0, time_point]
    """
    time_array = np.linspace(0.0, time_point, steps)
    return time_array


def find_nearest_neighbors(points, num_neighbors):
    """
    Find the nearest neighbors for each point in a given set of points.

    Parameters:
    -----------
    points : numpy.ndarray
        Array containing the coordinates of points with shape (N, 2),
        where N is the number of points and 2 is the dimensionality of the space.
    num_neighbors : int
        Number of nearest neighbors to find for each point.

    Returns:
    --------
    tuple
        A tuple containing two numpy.ndarray objects:
        - distances : Array of shape (N, num_neighbors) containing distances to the nearest neighbors.
        - indices : Array of shape (N, num_neighbors) containing indices of the nearest neighbors.

    Notes:
    ------
    The point has 0 index in indices array of the nearest neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute').fit(points)
    distances, indices = nbrs.kneighbors(points)
    return (distances, indices)


def cubic_spline_kernel_func(r, h):
    """
    Calculate the value of the cubic spline kernel function for a given distance and smoothing length.
    It assigns weights to neighboring particles based on their distance.

    Parameters:
    -----------
    r : numpy.ndarray
        Array of distances between particles.
    h : float
        Smoothing length.

    Returns:
    --------
    numpy.ndarray
        Array containing the values of the cubic spline kernel function for each distance.
    """
    k = r / h
    constant = 10.0 / (7.0 * np.pi * h ** 2)
    kernel = np.where(k < 1.0, constant * (1.0 - 1.5 * k ** 2 + 0.75 * k ** 3),
                      np.where(k < 2, 0.25 * constant * (2.0 - k) ** 3, 0.0))
    return kernel


def calculate_density(distances, masses, h):
    """
    Calculate the density at a given particle.

    Parameters:
    -----------
    distances : numpy.ndarray
        Array of distances between the particle and its neighboring particles.
    masses : numpy.ndarray
        Array of masses of neighboring particles.
    h : numpy.ndarray
        Smoothing length of neighboring particles.

    Returns:
    --------
    float
        Density at the particle of interest.

    Equation:
    ---------
    The density at a particle 𝜌(𝑟) is calculated as:
    𝜌(𝑟) = ∑_{𝑗=1}^{𝑁} m_{𝑗} * kernel(𝑟 - 𝑟_{𝑗}, h)
    where 𝑟 is the position vector of the particle of interest, 𝑟_{𝑗} is the position vector of the j-th neighboring particle,
    𝑚_{𝑗} is the mass of the j-th neighboring particle, kernel is the mean cubic spline kernel function for these two particles, 
    and h is the mean smoothing length.
    """
    total_density = np.sum(masses * (cubic_spline_kernel_func(distances, h[0]) + cubic_spline_kernel_func(distances, h)) / 2.0)
    return total_density


def calculate_pressure_from_energy(density, energy):
    """
    Calculate the pressure at a given particle. (w/o masses, only energies)
    """
    gamma = 1.67
    pressure = (gamma - 1.0) * np.abs(energy) * density ** (1.0 - gamma)
    return pressure


def sound_speed(pressure, density, gamma=1.67):
    """
    Calculate the sound of speed at a given particle.
    """
    return np.sqrt(gamma * pressure / density)


def get_pairwise_differences(array):
    """
    Calculate the paiwise difference for a couple of particles (for every pair in the array).
    pairwise_diffs[i,j] - the difference between v_{i} and v_{j}: (vx, vy)
    """
    pairwise_diffs = array[:, None, :] - array[None, :, :]
    return pairwise_diffs


def get_pairwise_mean(array):
    """
    Calculate the paiwise mean for a couple of particles (for every pair in the array).
    Use for smoothing lenght.
    """
    pairwise_mean = (array[:, None] + array) / 2.0
    return pairwise_mean


def get_neighbor_kernels(coords, h, i):
    """
    Calculate the mean of two kernels.
    """
    diffs = coords[i] - coords
    gradient_kernels = (get_gradient_kernel(diffs, h[i]) + get_gradient_kernel(diffs, h)) / 2
    return gradient_kernels


def get_c_matrix(pressure_arr, density_arr):
    """
    Calculate C coefficient for artifical viscosity calculation.
    """
    pressure_broadcasted = pressure_arr[:, None]
    density_broadcasted = density_arr[:, None]

    sound_speeds = sound_speed(pressure_broadcasted, density_broadcasted)
    return (sound_speeds + sound_speeds.T) / 2.0


def get_rho_matrix(density_arr):
    density_broadcasted = density_arr[:, None]

    return (density_broadcasted + density_broadcasted.T) / 2.0


def get_gradient_kernel(r, h):
    """
    Calculate gradient of the kernel. Return (grad_x, grad_y)
    """
    x, y = r[:, 0], r[:, 1]
    r_norm = np.linalg.norm(r, axis=1)
    k = r_norm / h
    C = 10.0 / (7.0 * np.pi)
    eps = 1e-9
    der_x = np.where(k == 0.0, 0.0, np.where(k < 1.0, C / h ** 4 * (2.25 * y * k - 3.0 * y),
                                             np.where(k <= 2.0, 3.0 * C * x * (2.0 - k) ** 2 / (4.0 * r_norm * h ** 3 + eps),
                                                      0.0)))

    der_y = np.where(k == 0.0, 0.0, np.where(k < 1.0, C / h ** 4 * (2.25 * x * k - 3.0 * x),
                                             np.where(k <= 2.0, 3.0 * C * y * (2.0 - k) ** 2 / (4.0 * r_norm * h ** 3 + eps),
                                                      0.0)))
    return np.column_stack((der_x, der_y))


def calculate_divergence(velocity, coords, density, h, mass, i, neighbor_kernels):
    """
    Calculate the divergence for xi in artifical viscosity calculation.
    """
    velocity_diffs = velocity - velocity[i]  ### v_i - v_j
    sum_terms = mass * np.sum(velocity_diffs * neighbor_kernels, axis=1)
    velocity_divergence = 1.0 / density * np.sum(sum_terms)
    return np.linalg.norm(velocity_divergence)


def calculate_curl(velocity, coords, density, h, mass, i, neighbor_kernels):
    """
    Calculate the curl for xi in artifical viscosity calculation.
    """
    velocity_diffs = velocity - velocity[i]  ### v_i - v_j
    cross_terms = mass * np.cross(velocity_diffs, neighbor_kernels)
    velocity_curl = 1.0 / density * np.sum(cross_terms, axis=0)
    return np.linalg.norm(velocity_curl)


def get_xi_coeff(h, pressure, densities, i, velocities, coords, masses, neighbor_kernels, delta2=1e-4):
    """
    Calculate coeff xi for artifical viscosity calculation.
    """
    с = sound_speed(pressure, densities)
    velocity_divergence = calculate_divergence(velocities, coords, densities, h, masses, i, neighbor_kernels)
    velocity_curl = calculate_curl(velocities, coords, densities, h, masses, i, neighbor_kernels)
    xi = velocity_divergence / (velocity_divergence + velocity_curl + delta2 * с / h[i])
    return xi


def get_xi_vec(h, pressure, densities, velocities, coords, masses, neighbor_kernels, delta2=1e-4):
    xi = np.zeros(h.shape[0])
    for i in range(h.shape[0]):
        xi[i] = get_xi_coeff(h, pressure[i], densities[i], i, velocities, coords, masses, neighbor_kernels, delta2)
    return xi


def get_mu(velocities, coords, h_array, delta1=1e-2):
    """
    Calculate coeff mu for artifical viscosity calculation.
    """
    diff_velocities = get_pairwise_differences(velocities)  ### v_i - v_j
    diff_coords = get_pairwise_differences(coords)
    h_mean = get_pairwise_mean(h_array)

    dot_product = np.sum(diff_velocities * diff_coords, axis=2)

    denominator = np.sum(diff_coords * diff_coords, axis=2) + delta1 * np.square(h_mean)
    mu = h_mean * dot_product / denominator
    return (mu, dot_product)


def get_artificial_viscosity_matrix(xi_vec, mu_matrix, c_matrix, rho_matrix, clauser_matrix, alpha=1, beta=0.5):
    artificial_viscosity = (xi_vec[:, np.newaxis] + xi_vec) * (beta * mu_matrix ** 2 - alpha * c_matrix * mu_matrix) / (
                2 * rho_matrix)
    artificial_viscosity[clauser_matrix >= 0] = 0.0
    return artificial_viscosity


def calculate_artificial_viscosity(velocities, coords, h, density, pressure, masses, neighbor_kernels):
    alpha = 1.0
    beta = 0.5
    delta1 = 1e-2
    delta2 = 1e-4
    c_matrix = get_c_matrix(pressure, density)
    rho_matrix = get_rho_matrix(density)
    mu_matrix, clauser_matrix = get_mu(velocities, coords, h, delta1)
    xi_vec = get_xi_vec(h, pressure, density, velocities, coords, masses, neighbor_kernels, delta2)
    artificial_viscosity = get_artificial_viscosity_matrix(xi_vec, mu_matrix, c_matrix, rho_matrix, clauser_matrix,
                                                           alpha, beta)
    return artificial_viscosity


def get_der_energy(coords, velocities, mass, h_array, density, pressure, viscosity, neighbor_kernels):
    N = mass.shape[0]
    i = 0
    energy = 0.0
    for j in range(N):
        vector = np.dot(velocities[i] - velocities[j], neighbor_kernels[j])
        energy += mass[j] * (pressure[i] / density[i] ** 2 + 0.5 * viscosity[i, j]) * vector
    if energy <= 0.0:
        energy = 1e-8
    return energy  # np.sum(energies)


def update_energy(energy, der_energy, dt):
    return energy + der_energy * dt


# ---------------------------------------------------------------------------------------------------

def calculate_gravitational_acceleration(coords, star_mass):
    G = 6.67430e-11

    x = coords[:, 0]
    y = coords[:, 1]

    distance_squared = x ** 2 + y ** 2
    distance = np.sqrt(distance_squared)

    mask = distance_squared != 0
    distance_cubed = np.where(mask, distance ** 3, 1)

    acceleration_x = - G * star_mass * x / distance_cubed
    acceleration_y = - G * star_mass * y / distance_cubed

    accelerations = np.column_stack((acceleration_x, acceleration_y))

    return accelerations


# ---------------------------------------------------------------------------------------------------

def calculate_momentum_eq(coords, mass, density, pressure, viscosity, acceleration, h_array, neighbor_kernels, energy):
    N = coords.shape[0]
    i = 0
    der_velocity = 0.0
    pres = calculate_pressure_from_energy(density.T[0], energy.T[0] / mass)
    for j in range(N):
        der_velocity += mass[j] * (pres[i] / density[i] ** 2 + pres[j] / density[j] ** 2 + viscosity[i, j]) * neighbor_kernels[j]
    res = -der_velocity + acceleration[i]
    return res


#
# def get_particles_res(point_index, indices_neighbors, distances, particle_coords, particle_velocities,
#                       particle_pressures, particle_densities, particle_masses, particle_smoothed_param,
#                       particle_energies, neighbor_kernels_mat, av_matrixes, new_particle_energies,
#                       new_particle_densities, new_particle_pressures, der_velocities, grav_acceleration):
#
#     neighbors_indices = indices_neighbors[point_index]
#     neighbors_distances = distances[point_index]
#
#     neighbors_coords = particle_coords[neighbors_indices]
#     neighbors_velocities = particle_velocities[neighbors_indices]
#     neighbors_pressures = particle_pressures[neighbors_indices]
#     neighbors_densities = particle_densities[neighbors_indices]
#     neighbors_masses = particle_masses[neighbors_indices]
#     neighbors_param = particle_smoothed_param[neighbors_indices]
#     neighbors_energies = particle_energies[neighbors_indices]
#
#     neighbor_kernels = get_neighbor_kernels(neighbors_coords, neighbors_param, 0)
#     neighbor_kernels_mat[point_index] = neighbor_kernels
#
#     av_matrix = calculate_artificial_viscosity(neighbors_velocities, neighbors_coords, neighbors_param,
#                                                neighbors_densities, neighbors_pressures, neighbors_masses,
#                                                neighbor_kernels)
#     av_matrixes[point_index] = av_matrix
#
#     der_energy = get_der_energy(neighbors_coords, neighbors_velocities, neighbors_masses, neighbors_param,
#                                 neighbors_densities, neighbors_pressures, av_matrix, neighbor_kernels)
#
#     new_particle_energies[point_index] = update_energy(particle_energies[point_index], der_energy, dt)
#     new_particle_densities[point_index] = calculate_density(neighbors_distances, neighbors_masses, neighbors_param)
#     new_particle_pressures[point_index] = calculate_pressure_from_energy(new_particle_densities[point_index],
#                                                                          new_particle_energies[point_index])
#
#     der_velocities[point_index, :] = calculate_momentum_eq(neighbors_coords, neighbors_masses,
#                                                            neighbors_densities, neighbors_pressures,
#                                                            av_matrixes[point_index], grav_acceleration,
#                                                            neighbors_param, neighbor_kernels_mat[point_index],
#                                                            neighbors_energies)
#

@measure_time
def run():
    num_particles = conf['particles']['num_particles']
    min_disk_radius = conf['particles']['min_disk_radius']
    max_disk_radius2 = conf['particles']['max_disk_radius2']
    max_shift = conf['particles']['max_shift']

    time_point = conf['time']['time_point']
    steps = conf['time']['steps']

    star_mass = conf['star']['star_mass']

    num_neigbors = conf['neigbors']['num_neigbors']

    particle_coords = get_init_coords_with_random_shift(num_particles, min_disk_radius, max_disk_radius2, max_shift)
    particle_velocities = get_init_velocities(particle_coords, star_mass)
    particle_masses = get_masses(num_particles)
    particle_smoothed_param = get_init_smoothed_param(num_particles)
    time_array = get_time_array(time_point, steps)

    particle_pressures = get_init_pressures(num_particles)
    particle_densities = get_init_densities(num_particles)
    particle_energies = get_init_energies(num_particles)

    time_array1 = time_array[:10]


    if 'data_test.txt' in os.listdir():
        os.remove('data_test.txt')

    counter = 0
    for dt in time_array1:
        print(counter)
        counter += 1

        distances, indices_neighbors = find_nearest_neighbors(particle_coords, num_neigbors)
        grav_acceleration = calculate_gravitational_acceleration(particle_coords, star_mass)

        new_particle_energies = np.zeros((num_particles))
        new_particle_densities = np.zeros((num_particles))
        new_particle_pressures = np.zeros((num_particles))
        der_velocities = np.zeros((num_particles, 2))

        av_matrixes = np.zeros((num_particles, num_neigbors, num_neigbors))
        neighbor_kernels_mat = np.zeros((num_particles, num_neigbors, 2))

        if (particle_pressures < 0).any():
            breakpoint()

        # def get_step_res(point_index):
        #     return get_particles_res(point_index, indices_neighbors, distances, particle_coords, particle_velocities,
        #               particle_pressures, particle_densities, particle_masses, particle_smoothed_param,
        #               particle_energies, neighbor_kernels_mat, av_matrixes, new_particle_energies,
        #               new_particle_densities, new_particle_pressures, der_velocities, grav_acceleration)

        # with Pool() as pool:
        #     pool.map(get_step_res, range(num_particles))


        for point_index in range(num_particles):
            neighbors_indices = indices_neighbors[point_index]
            neighbors_distances = distances[point_index]

            neighbors_coords = particle_coords[neighbors_indices]
            neighbors_velocities = particle_velocities[neighbors_indices]
            neighbors_pressures = particle_pressures[neighbors_indices]
            neighbors_densities = particle_densities[neighbors_indices]
            neighbors_masses = particle_masses[neighbors_indices]
            neighbors_param = particle_smoothed_param[neighbors_indices]
            neighbors_energies = particle_energies[neighbors_indices]


            neighbor_kernels = get_neighbor_kernels(neighbors_coords, neighbors_param, 0)
            neighbor_kernels_mat[point_index] = neighbor_kernels

            av_matrix = calculate_artificial_viscosity(neighbors_velocities, neighbors_coords, neighbors_param,
                                                       neighbors_densities, neighbors_pressures, neighbors_masses,
                                                       neighbor_kernels)
            av_matrixes[point_index] = av_matrix

            der_energy = get_der_energy(neighbors_coords, neighbors_velocities, neighbors_masses, neighbors_param,
                                        neighbors_densities, neighbors_pressures, av_matrix, neighbor_kernels)


            new_particle_energies[point_index] = update_energy(particle_energies[point_index], der_energy, dt)
            new_particle_densities[point_index] = calculate_density(neighbors_distances, neighbors_masses, neighbors_param)
            new_particle_pressures[point_index] = calculate_pressure_from_energy(new_particle_densities[point_index],
                                                                                 new_particle_energies[point_index])

            der_velocities[point_index, :] = calculate_momentum_eq(neighbors_coords, neighbors_masses,
                                                                   neighbors_densities, neighbors_pressures,
                                                                   av_matrixes[point_index], grav_acceleration,
                                                                   neighbors_param, neighbor_kernels_mat[point_index], neighbors_energies)

        particle_coords += dt * particle_velocities
        particle_velocities += dt * der_velocities

        particle_energies = new_particle_energies
        particle_densities = new_particle_densities
        particle_pressures = new_particle_pressures


        if (particle_pressures < 0).any():
            breakpoint()

        with open("data_test.txt", "a") as file:
            file.write("# Time : {}\n".format(dt))
            for coords in particle_coords:
                file.write(" ".join(map(str, coords)) + "\n")


        def draw(array, counter, title, directory):
            y_values = array
            x_values = np.linspace(0, 100, len(y_values))

            plt.figure()

            plt.scatter(x_values, y_values, color='gray', s=1)

            plt.title(f'Распределение точек: {title}')
            plt.xlabel('X')
            plt.ylabel('Y')

            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(os.path.join(directory, f'{title}_{counter}.png'))
            plt.clf()

        draw(particle_energies, counter, 'energies', 'energies_directory')
        plt.close()
        draw(particle_densities, counter, 'densities', 'densities_directory')
        plt.close()
        draw(particle_pressures, counter, 'pressures', 'pressures_directory')
        plt.close()
        


run()
