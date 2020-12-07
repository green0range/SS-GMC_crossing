# coding: utf8

import rebound
import numpy as np
import matplotlib.pyplot as plt
import time as time_
import os
from scipy import optimize
import data
import external
import timeit

class CustomException(Exception):
    """Allows for a custom exception to be raised."""
    pass

# We have have a global pointer to the simulation particle list for the external forces method.
particles = None
number_of_simulation_objects = 0
object_metadata = None
enable_external_forces = None
molecular_cloud = None
random = np.random.RandomState()


def read_in_barycenter_data(file):
    """Reads in planetary data, calculates and returns the barycenter"""
    assert os.path.exists(file)
    f = open(file)
    pos = []
    vel = []
    mass = []
    n = int(f.readline())
    for i in range(n):
        mass.append(float(f.readline().split()[0]))
        pos.append(f.readline().split())
        vel.append(f.readline().split())
    f.close()
    print(mass)
    print(pos)
    print(vel)
    bc_x, bc_y, bc_z = 0, 0, 0
    bc_xv, bc_yv, bc_zv = 0, 0, 0
    for i in range(n):
        bc_x += mass[i] * float(pos[i][0])
        bc_y += mass[i] * float(pos[i][1])
        bc_z += mass[i] * float(pos[i][2])
        bc_xv += mass[i] * float(vel[i][0])
        bc_yv += mass[i] * float(vel[i][1])
        bc_zv += mass[i] * float(vel[i][2])

    bc_pos = [bc_x/sum(mass), bc_y/sum(mass), bc_z/sum(mass)]
    bc_vel = [bc_xv/sum(mass), bc_yv/sum(mass), bc_zv/sum(mass)]
    return bc_pos, bc_vel, sum(mass)/mass[0]

def read_in_n_xyx_xvyvzv_data(file, n):
    """The reads data which is in the format particle_number \n x y z \n xv yv zv \n next_particle_number \n etc

    where the particles must be ordered by particle number.

    Args:
        file: path to file to be read in.
        n: size of sample to return. If n > number of particles in the file, it will just return all the particles.
    """
    assert os.path.exists(file)
    print("Opening file", file)
    f = open(file)
    tp_position = []
    tp_velocity = []
    count = 1
    expect_position = False
    expect_velocity = False
    for line in f.readlines():
        line = line.strip(" \r\n")
        if expect_position:
            ls = line.split()
            if len(ls) == 3:
                tp_position.append([float(ls[0]), float(ls[1]), float(ls[2])])
                expect_velocity = True
                expect_position = False
        elif expect_velocity:
            ls = line.split()
            if len(ls) == 3:
                tp_velocity.append([float(ls[0]), float(ls[1]), float(ls[2])])
                expect_velocity = False
        elif int(line) == count:  # if the file is out of order this will fail and avoid reading bad data.
            expect_position = True
            count += 1
        else:
            print("Warning, did not find particle number after count", count, ". File may be unordered or corrupt")
    f.close()
    print("read", count, "test particles")
    assert len(tp_velocity) == count - 1
    assert len(tp_position) == count - 1
    # Now I need to convert units. The positions are already in au. velocities are in au/yr
    for i in range(len(tp_velocity)):
        for j in range(3):
            tp_velocity[i][j] /= (2*np.pi)
    # now subsample the data to have n test particles:
    if n < count - 1:
        # create an array indices from which we randomly select indices to use.
        indicies = np.arange(0, len(tp_position))
        selection = random.choice(indicies, size=n, replace=False)
        tp_p_selection = []
        tp_v_selection = []
        for s in selection:
            tp_p_selection.append(tp_position[s])
            tp_v_selection.append(tp_velocity[s])
        return tp_p_selection, tp_v_selection
    else:
        return tp_position, tp_velocity

class Sim:
    def __init__(self, sim_length, solar_system="solar_systems/sun.bin"):
        """ Constructor method, does setup.
        Args:
            sim_length: length of the simulation in years
            solar_system: Base solar system to use. 4 are provided in folder "solar_systems"
        """
        self.comet_mass_density = 0
        self.asteroid_mass_density = 0
        if "_barycenter_" not in solar_system:
            self.s = rebound.Simulation(solar_system)
        else:
            self.s = rebound.Simulation()
            self.s.units = ("solarmass", "au", "yr2pi")
        """ The next 2 lines deal with units:
        Firstly, we use natural units: this means that G = 1.
        Next we make sure rebound doesn't calculate collisions. Collisions should be infrequent enough not to
        cause any problems, since Oort cloud objects are spaced far apart. Not checking collisions means we can
        set the s.particles[].r to a chosen radius in meters without rebound thinking it is in au. (because it
        doesn't check collisions)
        """
        #self.s.units = ("solarmass", "au", "yr2pi") # This cannot be set since we are loading in an exsiting sim.
        self.s.collision = "none"
        self.number_of_orbit_plot_frames = 200
        self.ps = self.s.particles
        self.year = 2. * np.pi
        self.heartbeat_previous_t = 0
        self.u = data.Unit_Conversion()
        self.integration_time = sim_length/self.year
        self.heartbeat = self.are_you_okay
        self.n = 0  # number of comets
        self.start_time = time_.time()
        self.percent_asteroid = 0
        self.object_metadata = []

        self._are_you_okay_t = 0

    def load_binary(self, file):
        global object_metadata, particles, number_of_simulation_objects
        self.s = rebound.Simulation(file)
        self.n = len(self.s.particles)-1
        self.ps = self.s.particles
        particles = self.ps
        number_of_simulation_objects = self.n
        object_metadata = self.object_metadata
        for i in range(self.n):
            self.object_metadata.append(["unknown", 0, 0, 0])
            object_metadata = self.object_metadata
        self.integration_time += self.s.t  # if we load a binary, the sim time is probs not 0

    def set_densities(self, comet, asteroid, percent_asteroid):
        assert percent_asteroid >= 0
        assert percent_asteroid <= 1
        self.comet_mass_density = comet
        self.asteroid_mass_density = asteroid
        self.percent_asteroid = percent_asteroid

    @staticmethod
    def set_random_state_seed(seed):
        """ Initialises the random object to use the given seed such that resulting Oort cloud can be replicated.
        :param seed:
        :return:
        """
        global random
        random = np.random.RandomState(seed)

    @staticmethod
    def get_random_distribution(n, min, max):
        """ Generates a uniform random distribution is range [min, max)
        Args:
            n: Size of distribution.
            min: Minimum value (inclusive)
            max: Maximum value (exclusive)
        """
        global random
        return (max - min) * random.random_sample(n) + min

    @staticmethod
    def get_eccentricity_distribution(n):
        global random
        e_potential = np.linspace(0, 0.9996, n)
        N_e = np.zeros(n)
        for i in range(len(e_potential)):
            N_e[i] = 29.4*np.exp(-1*((e_potential[i]-0.945)**2)/(2*0.0019)) + 4.8*np.exp(-1*((e_potential[i]-0.945)**2)/(2*0.1475))
        N_e = N_e/sum(N_e)
        return random.choice(e_potential, size=n, replace=True, p=N_e)

    @staticmethod
    def get_a_distribution(e, min_, max_, min_pericenter):
        """ The selects a uniform distribution between min and max, except for cases where the pericenter would be
        too close to the sun. In such cases it finds a new minimum such that the pericenter requirement is satisfied"""
        global random
        a_min = max(((1+e)/(1-e**2)) * min_pericenter, min_)
        return (max_ - a_min) * random.rand() + a_min

    @staticmethod
    def get_size_distribution(n, min, max, distribution):
        """
        :param min:
        :param max:
        :param distribution:
        :return: The radius in km
        """
        global random
        weight = []
        radius = []
        if type(distribution) == data.Size_Distribution:
            if distribution.get_uniform():
                return random.uniform(min, max, n)
            increment = (max - min) / (10*n)
            for r in np.arange(min, max, increment):
                N = distribution.get_number(r) - distribution.get_number(r+increment)
                if N > 0:
                    radius.append(r)
                    weight.append(N)
            A = sum(weight)
            for i in range(len(weight)):
                weight[i] /= A
            return random.choice(radius, n, p=weight)
        if distribution == "meech-hainaut-marsden":
            # This is for backwards compatibility
            d = data.Size_Distribution(default_q=1.45)
            return Sim.get_size_distribution(n, min, max, d)
        else:
            raise CustomException("Distribution type not defined")

    @staticmethod
    def get_mass(radius, density):
        """calculates the mass of a constant density spherical object."""
        return (4/3)*np.pi*(radius**3)*density

    @staticmethod
    def kg_to_solmass(m):
        return m / 1.98847e30

    def import_n_xyz_xvyvzv_sample(self, f, f2, n, min_size, max_size, size_dist_type, min_a):
        """ This imports data in the format of David's formation simulation results.

        I also corrects the data by:
         - removing particles with a < a_min, eg to remove the scattered disk
         - barycentric corrections
        """
        global particles, number_of_simulation_objects, object_metadata
        self.n = n
        pos, vel = read_in_n_xyx_xvyvzv_data(f, n)
        if f2 is not None:  # set f2 to None if already barycentric
            barycenter, barycenter_v, barycenter_M = read_in_barycenter_data(f2)
            '''Perform the barycentric correction'''
            for i in range(len(pos)):
                pos[i][0] -= barycenter[0]
                pos[i][1] -= barycenter[1]
                pos[i][2] -= barycenter[2]
                vel[i][0] -= barycenter_v[0]
                vel[i][1] -= barycenter_v[1]
                vel[i][2] -= barycenter_v[2]
        else:
            barycenter_M = 1.0013358462653765
        # adds the barycenter mass
        self.s.add(m=barycenter_M, x=0, y=0, z=0, vx=0, vy=0, vz=0)
        comet_sample_sizes = self.get_size_distribution(n, min_size, max_size, size_dist_type)
        comet_sample_mass = []
        for i in range(len(comet_sample_sizes)):
            if i <= n * self.percent_asteroid:
                # Label these as asteroids
                comet_sample_mass.append(self.kg_to_solmass(self.get_mass(comet_sample_sizes[i] * 1000,
                                                                          self.asteroid_mass_density)))  # r: km->m
                self.object_metadata.append(["asteroid", comet_sample_sizes[i], self.asteroid_mass_density,
                                             self.get_mass(comet_sample_sizes[i] * 1000, self.asteroid_mass_density)])
            else:
                comet_sample_mass.append(self.kg_to_solmass(self.get_mass(comet_sample_sizes[i] * 1000,
                                                                          self.comet_mass_density)))  # r: km->m
                self.object_metadata.append(["comet", comet_sample_sizes[i], self.comet_mass_density,
                                             self.get_mass(comet_sample_sizes[i] * 1000, self.comet_mass_density)])
        for i in range(len(pos)):
            self.s.add(m=comet_sample_mass[i], x=pos[i][0], y=pos[i][1], z=pos[i][2], vx=vel[i][0], vy=vel[i][1], vz=vel[i][2], r=comet_sample_sizes[i]*1000)
        self.s.move_to_com()
        self.ps = self.s.particles
        '''Remove particles with a < a_min'''
        marked_for_removal = []
        for i in range(1, len(self.s.particles)):
            if self.s.particles[i].a < min_a or self.s.particles[i].a >= 5000000:
                marked_for_removal.append(self.s.particles[i].hash)  # removing immediately would upset the indexing
            if self.s.particles[i].e > 1:
                marked_for_removal.append(self.s.particles[i].hash)
        for i in range(len(marked_for_removal)):
            self.s.remove(hash=marked_for_removal[i])
        particles = self.ps
        # this is the number of comets, so we exclude the barycenter object.
        self.n = len(particles) - 1
        number_of_simulation_objects = self.n
        number_of_simulation_objects = self.n
        object_metadata = self.object_metadata

    def generate_comet_sample(self, n, min_size, max_size,
                              min_a, max_a, min_e, max_e,
                              min_i, max_i, min_omega, max_omega,
                              min_Omega, max_Omega, min_tau, max_tau, size_dist_type):  # note, tau should be
        global particles, number_of_simulation_objects, object_metadata
        # between 0 and 1,
        # as a percent of period T.
        """Generate a random sample of comets.

        Args:
            n: Number of comets
            min_size: minimum mass of comet
            max_size: maximum mass of comet
            min_a: min semimajor axis
            max_a: max semimajor axis
            min_e: min eccentricity
            max_e: max "
            min_i: min inclination
            max_i: max "
            min_omega: min argument of pericentre
            max_omega: Max " " "
            min_Omega: Min longitude of the ascending node
            max_Omega: Max " " " " "
            min_tau: Min time of perihelion, as fraction of period
            max_tau: Max " " " " " " "
        """
        # Check everything is within relevant bounds
        for minmax in [[min_size, max_size], [min_a, max_a], [min_e, max_e], [min_omega, max_omega],
                       [min_Omega, max_Omega], [min_tau, max_tau], [min_i, max_i]]:
            assert minmax[0] < minmax[1]
        assert n > 0
        assert min_tau >= 0 and max_tau <= 1
        assert min_a >= 0
        assert min_e >= 0 and max_e <= 1  # for bound population
        assert min_i >= 0 and max_i <= np.pi
        assert min_omega >= 0 and max_omega <= 2*np.pi
        assert min_Omega >= 0 and max_Omega <= 2*np.pi

        self.n = n
        comet_sample_sizes = self.get_size_distribution(n, min_size, max_size, size_dist_type)
        comet_sample_mass = []
        for i in range(len(comet_sample_sizes)):
            if i <= n*self.percent_asteroid:
                # Label these as asteroids
                comet_sample_mass.append(self.kg_to_solmass(self.get_mass(comet_sample_sizes[i]*1000,
                                                                          self.asteroid_mass_density)))  # r: km->m
                self.object_metadata.append(["asteroid", comet_sample_sizes[i], self.asteroid_mass_density,
                                             self.get_mass(comet_sample_sizes[i]*1000, self.asteroid_mass_density)])
            else:
                comet_sample_mass.append(self.kg_to_solmass(self.get_mass(comet_sample_sizes[i]*1000,
                                                                          self.comet_mass_density)))  # r: km->m
                self.object_metadata.append(["comet", comet_sample_sizes[i], self.comet_mass_density,
                                             self.get_mass(comet_sample_sizes[i] * 1000, self.comet_mass_density)])
        #comet_sample_a = self.get_random_distribution(n, min_a, max_a)
        comet_sample_e = self.get_eccentricity_distribution(n)  #self.get_random_distribution(n, min_e, max_e)
        comet_sample_omega = self.get_random_distribution(n, min_omega, max_omega)
        comet_sample_Omega = self.get_random_distribution(n, min_Omega, max_Omega)
        comet_sample_tau = self.get_random_distribution(n, min_tau, max_tau)
        comet_sample_inc = self.get_random_distribution(n, min_i, max_i)
        comet_sample_a = []
        for i in range(n):
            comet_sample_a.append(self.get_a_distribution(comet_sample_e[i], min_a, max_a, 40))

        for i in range(n):
            # calculate T
            T = np.sqrt((4*np.pi**2*comet_sample_a[i]**3)/(1))
            self.s.add(m=comet_sample_mass[i],
                    a=comet_sample_a[i],
                    e=comet_sample_e[i],
                    inc=comet_sample_inc[i],
                    omega=comet_sample_omega[i],
                    Omega=comet_sample_Omega[i],
                    T=comet_sample_tau[i]*T,
                    r=comet_sample_sizes[i]*1000)  # record r in m, rebound doesn't use r in it's integrations.

        self.s.move_to_com()
        self.ps = self.s.particles
        particles = self.ps
        number_of_simulation_objects = self.n
        object_metadata = self.object_metadata

    def enable_external_forces(self, drag, plummer, tide):
        """

        :param drag:
        :param plummer:
        :param tide:
        :return:
        """
        global enable_external_forces
        enable_external_forces = (drag, plummer, tide)
        if drag or plummer or tide:
            self.s.additional_forces = self.external_forces
            self.s.force_is_velocity_dependent = 1

    @staticmethod
    def external_forces(reb_sim):
        """Provide external forces

        Args:
            reb_sim: Rebound simulation to apply external forces on.
        """
        global particles, molecular_cloud
        mc = molecular_cloud
        u = data.Unit_Conversion()  # create a unit conversion object, everything must be converted to SI
        #t1 = time_.time_ns()
        for i in range(len(particles)):
            '''For each particle we find the external force to apply'''
            velocity = np.array([u.vel_to_SI(particles[i].vx),
                                u.vel_to_SI(particles[i].vy),
                                u.vel_to_SI(particles[i].vz)])
            #print(velocity)
            #print(u.time_to_SI(reb_sim.contents.t))

            if enable_external_forces[0]:
                F_drag = mc.get_drag_force(velocity, 2*np.pi*particles[i].r, particles[i].x,
                                  particles[i].y,
                                  particles[i].z, u.time_to_SI(reb_sim.contents.t))
                if np.any(np.isnan(F_drag)):
                    F_drag = np.array([0,0,0])
            else: F_drag = np.array([0,0,0])
            acceleration = F_drag/u.mass_to_SI(particles[i].m)
            if enable_external_forces[1]:
                g_plummer = mc.plummer_sphere_g(particles[i].x, particles[i].y, particles[i].z, u.time_to_SI(reb_sim.contents.t))
                g_plummer = u.acc_from_SI(g_plummer)
                if np.any(np.isnan(g_plummer)):
                    g_plummer = np.array([0,0,0])
            else: g_plummer = np.array([0,0,0])
            if enable_external_forces[2]:
                g_tide = mc.galactic_tide(particles[i].z, particles[i].x)
                if np.isnan(g_tide):
                    g_tide = 0
            else: g_tide = 0
            '''Debugging messages'''
            #print("Drag force on particle ", str(i), "is", str(acceleration), "N/kg")
            #print("Plummer sphere gravity on particle", str(i), "is", g_plummer, "N/kg")
            #print("Galactic tide force on particle", str(i), "is", g_tide, "z hat, N/kg")
            particles[i].ax += u.acc_from_SI(acceleration[0]) + g_plummer[0]
            particles[i].ay += u.acc_from_SI(acceleration[1]) + g_plummer[1]
            particles[i].az += u.acc_from_SI(acceleration[2]) + g_plummer[2] + g_tide
        #t2 = time_.time_ns()
        #print("Timer: external forces:", str(t2 - t1), "nanoseconds")

    def are_you_okay(self, s):
        """Heartbeat function, run at each integration step. It prints the simulation progress and
        remaining time estimates.

        Args:
            s: simulation
        """
        self._are_you_okay_t = s.contents.t*self.year
        if self._are_you_okay_t > (self.heartbeat_previous_t + (self.integration_time / 1000)):
            percent_done = 100 * self._are_you_okay_t / (self.integration_time*self.year)
            runtime = time_.time() - self.start_time
            remaining_time = (100 - percent_done) * (runtime / percent_done)
            print("Sim time:", str(self._are_you_okay_t), "/", str(self.integration_time*self.year), "yr runtime:",
                  runtime, "s remaining time:", remaining_time, "s")
            self.heartbeat_previous_t = self._are_you_okay_t

    @staticmethod
    def plot_size_distribution(n, min, max, dist):
        """This plots a size distribution. It is not needed for running the simulation, just a visualisation tool"""
        c = Sim.get_size_distribution(n, min, max, dist)
        #print(c)
        #plt.hist(c, bins=int(n/100))
        plt.scatter(c, np.ones(n))
        #plt.yscale('log')
        #plt.xscale('log')
        plt.show()

    def remove_ejected_particles(self):
        """ DON'T USE: NOT WORKING

        Checks for ejected particles are removes them from the simulation

        param: force_remove: if True it will always remove a particle, even if not ejected. This is to test the removal
        process.

        returns: removed indices: this is so that the particle_data list can be updated to skip these indices
        """
        h = []
        num_major_particles = len(self.s.particles) - self.n
        for i, p in enumerate(self.s.particles[num_major_particles:]):
            if p.a > 500000 or p.a < 0:
                h.append(p.hash)
        for i in range(len(h)):
            self.s.remove(hash=h[i])
        print("removed", str(len(h)), "particles")
        return h


    def integrate_simulation(self, integration_step=0.1, number_of_datapoints=10000, integrator="IAS15",
                             dry_run=False):
        """ Integrates the simulation.

        Args:
            integration_step: integration step used by integrator
            integration_step: integration step used by integrator
            number_of_datapoints: Number of datapoints to collect and save to file during the integration
            integrator: rebound integrator to use.
            dry_run: If True, it only pretends to run the simulation, with much reduced particle count and short
            time. For testing only.

        Returns: list of Particle.
        """
        self.s.move_to_com()
        self.s.dt = integration_step
        self.s.integrator = integrator
        if integrator == "WHFAST":
            print("Warning, this shouldn't be used for non-conservative forces")
            self.s.ri_whfast.safe_mode = 0
        if integrator == "IAS15":
            self.s.ri_ias15.epsilon = 0  # this turns off the adaptive timestep
        particle_data = []
        particle_labels = []
        # self.s.status()
        print("###")
        print("REBOUND version: "+rebound.__version__)
        print("Number of particles: "+str(len(self.s.particles)))
        print("Selected integrator: "+self.s.integrator)
        print("Current timestep: "+str(self.s.dt))
        print("###")
        """The maximum number of files we can open at once is 1024, so here I split the file
        writing tasks in batches of 1000 files at a time. If n < max_batch size the function
        doesn't need to close the files until the end of the simulation and constantly adds to
        them.
        If n exceeds (or is =) then we write 1000 file headers, close those files, open the next
        next batch, repeat.
        """
        num_major_particles = len(self.s.particles) - self.n
        if dry_run:
            max_batch_size = 5
        else:
            max_batch_size = 1000
        files_to_open = self.n
        file_batches = int(self.n/max_batch_size) + 1
        print("files to open " + str(files_to_open))
        for i in range(len(self.s.particles)):
            self.s.particles[i].hash = i
            if i >= num_major_particles:
                particle_data.append(data.Particle(label=str(self.s.particles[i].hash)))
                particle_data[i-num_major_particles].set_metadata(self.object_metadata[i-num_major_particles][0],
                                                                  self.object_metadata[i-num_major_particles][1],
                                                                  self.object_metadata[i-num_major_particles][2],
                                                                  self.object_metadata[i-num_major_particles][3])
                particle_data[i-num_major_particles].open_stream('w')
                particle_data[i-num_major_particles].close_stream()
        '''
        self.ps = self.s.particles
        for j in range(file_batches):
            if self.n >= max_batch_size:  # this means if n == 1000, we get a batch of 1000 and an empty batch.
                files_to_open = np.minimum(max_batch_size, self.n-(j*max_batch_size))
            for i in range(files_to_open):
                particle_data.append(data.Particle(label=str(self.ps[i+j*max_batch_size - num_major_particles].hash)))
                particle_labels.append(str(self.ps[i+j*max_batch_size - num_major_particles].hash))
                particle_data[i+j*max_batch_size].set_metadata(self.object_metadata[i+j*max_batch_size][0],
                                                               self.object_metadata[i+j*max_batch_size][1],
                                                               self.object_metadata[i+j*max_batch_size][2],
                                                               self.object_metadata[i+j*max_batch_size][3],
                                                                  label=particle_labels[i+j*max_batch_size])
                particle_data[i+j*max_batch_size].open_stream('w', label=particle_labels[i+j*max_batch_size])
                if file_batches != 1:
                    particle_data[i+j*max_batch_size].close_stream()
        '''
        self.start_time = time_.time()
        self.s.heartbeat = self.are_you_okay
        # This is the timestep between writing the results out to the file, not the timestep used by
        # the integrator.
        dt = self.integration_time/number_of_datapoints
        # putting the start time just before integrations will give a better estimate of the time remaining.
        self.start_time = time_.time()
        num_ejec_checks = 10
        ejec_checks_done = 0
        print(particle_data[0].get_label())
        for time in np.arange(0, self.integration_time, dt):
            self.s.integrate(time, exact_finish_time=0)
            #if time > (self.integration_time/num_ejec_checks) * (ejec_checks_done+1):
            #    self.remove_ejected_particles()
            #    ejec_checks_done += 1
            self.ps = self.s.particles
            files_to_open = self.n
            # print("files to open "+str(files_to_open))
            file_batches = int(self.n / max_batch_size) + 1
            #t1 = time_.time_ns()
            for i in range(num_major_particles, len(self.ps)):
                particle_data[0].open_stream('a', label=str(self.ps[i].hash))
                particle_data[0].save_next(self.ps[i].a, self.ps[i].e, self.ps[i].inc, time*self.year, self.ps[i].x, self.ps[i].y, self.ps[i].z, label=str(self.ps[i].hash))

            particle_data[0].close_stream()
            '''
            for j in range(file_batches):
                if self.n >= max_batch_size:
                    files_to_open = np.minimum(max_batch_size, self.n - (j * max_batch_size))
                for i in range(files_to_open):
                    #assert particle_data[i + j * max_batch_size][1] == self.ps[i + j * max_batch_size - len(
                    # removed_particle_indices)].hash
                    access = num_major_particles + i + j * max_batch_size
                    for k in range(len(removed_particle_indices)):
                        if access <= removed_particle_indices[k]:
                            break
                        else:
                            access -= 1
                        print(access)
                    if file_batches != 1:
                        particle_data[i+j*max_batch_size][0].open_stream('a', label=particle_data[i+j*max_batch_size][1])
                    #print("label:"+str(particle_data[i+j*max_batch_size][1]), "index:"+str(i+j*max_batch_size), "accessing:"+str(access))
                    # assert self.ps[access].hash == particle_data[i+j*max_batch_size][2]
                    particle_data[i+j*max_batch_size][0].save_next(self.ps[access].a, self.ps[access].e, self.ps[access].inc, time*self.year, self.ps[access].x, self.ps[access].y, self.ps[access].z, label=particle_data[i+j*max_batch_size][1])
                    if file_batches != 1:
                        particle_data[i+j*max_batch_size][0].close_stream()
            #t2 = time_.time_ns()
            #print("Timer: main loop, write data:", str(t2 - t1), "nanoseconds")
            '''
        if file_batches == 1:
            for i in range(self.n):
                particle_data[i].close_stream()
        return particle_data


    def save_current_simulation_state(self, file):
        self.s.save(file)


#d = data.Size_Distribution(default_q="uniform")
#c = Sim.plot_size_distribution(1000, 0.1, 100, d)
