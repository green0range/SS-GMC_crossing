# coding: utf8

import sim
import data
import numpy as np
import time as time_
import sys
import os
import external
import shutil

end = False
version = "Alpha 11.08.20"

class Switch:
    """Provides a `switch` statements for code clarity (rather than using if...elifs.)
    """
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False # Allows a traceback to occur

    def __call__(self, *values):
        return self.value in values


def plot_cmd(uin):
    """ User interactive plotting tool.
    Args:
        uin:
    """
    type = input("type of plot? [(aei)/orbital]")
    if type == "aei" or type == "":
        location = input("Location of particle data:")
        uuids = input("Particle UUIDs/labels in space [ ] separated list:").split(' ')
        p = data.Plotter(uuids, location=location)
        options = input("Plotting options (leave blank for default):")
        save_as = None
        show = True
        gradient = True
        colourmap = "viridis"
        alim = None
        for opt in options:
            o = opt.split("=")
            if o[0] == "save_as":
                save_as = o[1]
            if o[0] == "show":
                show = bool(o[1])
            if o[0] == "gradient":
                gradient = o[1]
            if o[0] == "colourmap":
                colourmap = o[1]
            if o[0] == "alim":
                alim = (float(o[1].split("-")[0]), float(o[1].split("-")[1]))
        p.plot_aei(save_as=save_as, show=show, gradient=gradient, colourmap=colourmap, alim=alim)
    elif type == 'orbital':
        location = input("Location of particle data:")
        uuids = input("Particle UUIDs/labels in space [ ] separated list:").split(' ')
        p = data.Plotter(uuids, location=location)
        p.plot_orbit(500, 50)


def simulate_cmd(uni):
    """Interactive simulate command. ToDo: this is not updated to use new parameters as of 10 May.
    Args:
        uni:
    """
    save_path = ("Sim" + str(time_.localtime().tm_mday) +
                 "-" + str(time_.localtime().tm_mon) +
                 "-" + str(time_.localtime().tm_year) +
                 "_" + str(time_.localtime().tm_hour) +
                 ":" + str(time_.localtime().tm_min))
    user_input = input("Save simulation as (default="+save_path+"):")
    if not user_input == "":
        save_path = user_input
    path_number = 0
    try:
        while os.path.exists(save_path):  # This only occurs if multiple runs occur in the same minute.
            path_number += 1
            save_path += "_" + str(path_number)
        os.mkdir(save_path)
    except Exception:
        print("Invalid file path, exiting simulation prompt")
        return
    try:
        n = int(input("Number of particles:"))
        sim_length = int(input("Simulation length in earth years:"))
        generation_params = input("in a space [' '] separated list please provide:\n"+
                                  "min_M max_M min_a max_a min_e max_e min_i max_i min_omega "+
                                  "max_omega min_Omega max_Omega min_tau max_tau\n"+
                                  "or just press enter to use default options:")
    except Exception:
        print("Invalid input, exiting simulation prompt")
        return
    s = sim.Sim(sim_length)
    if generation_params == "":
        s.generate_comet_sample(n, 1e-20, 1e-15,  # num, mass
                            25, 50,  # a
                            0, 1,  # e
                            0, np.pi,  # i
                            0, 2 * np.pi,  # omega (angle from ascending node to x hat) 'argument of pericentre'
                            0, 2 * np.pi,  # OMEGA 'Longitude of ascending node' (Reference line to ascending node)
                            0, 1.0)  # tau, time of pericentre - see note above
    else:
        arg = generation_params.split(' ')
        try:
            s.generate_comet_sample(int(n), float(arg[0]), float(arg[1]), float(arg[2]), float(arg[3]), float(arg[4]),
                                    float(arg[5]), float(arg[6]), float(arg[7]), float(arg[8]), float(arg[9]),
                                    float(arg[10]), float(arg[11]), float(arg[12]), float(arg[13]), float(arg[14]))
        except IndexError:
            print("Not enough parameters provided, exiting simulation prompt")
            return
    result = s.integrate_simulation()
    p = data.Plotter(result)
    p.plot_aei(save_as=os.path.join(save_path, "aei_plot.png"), show=True, alim=(0, 60))
    for i in range(len(result)):
        result[i].save_csv(save_path)


def help_cmd():
    """Print out some usage examples and documentation. ToDo: write the help statements.
    """
    print("Please read the documentation. Contact wsa42@uclive.ac.nz if unclear.")
    print("Tips: all commands should be lower case, try key words like 'plot gmc density',\n 'plot sim path/to/data', 'generate (gmc/sizedist)' find (gmc mass/sizedist), etc.")
    print("To exit, type 'exit' or 'q' or 'quit'")

class Oort_data:
    """Reads an oort.dat file and runs a simulation based in its parameters

    the oort.dat should be formatted in the following way:
        # any line containing a hash will be ignored
        # any parameter that doesn't exist will also be ignored
        parameter=value
        new_parameter_on_new_line=value2
        parameter3=value3 # if there is a hash anywhere on the line, that whole line is ignored
        # Let's try set parameter3 again
        parameter3=value3
        # OK now it is set.

    To see a list of valid parameter see the set_value method.
    """

    def __init__(self, f):
        self.cloud_inner_radius = 0
        self.cloud_outer_radius = 0
        self.population = 0
        self.simulation_length = 0
        self.asteroid_mass_density = 0
        self.maximum_comet_size = 0
        self.minimum_comet_size = 0
        self.minimum_asteroid_size = 0
        self.maximum_asteroid_size = 0
        self.comet_mass_density = 0
        self.number_of_orbit_plots = 0
        self.create_gif_from_orbit_plots = False
        self.number_of_datapoints = 0
        self.integrator_timestep = 0.1
        self.integrator = "IAS15"
        self.size_distribution = ""
        self.solar_system_file = "solar_systems/sun.bin"
        self.percent_asteroid = 0
        self.enable_drag = True
        self.enable_plummer_gravity = True
        self.enable_galactic_tide = True
        self.set_timestep_by_crossing_time = 0
        self.ism_density = 0
        self.gmc_peak_density = 0
        self.gmc_velocity_dispersion = 0
        self.use_crossing_time_as_simulation_length = False
        self.condensations = []
        self.velocity_of_gmc = np.array([0, 0, 0])
        self.velocity_of_sun = np.array([0, 0, 0])
        self.solar_system_r_0_y = 0
        self.create_path_plot = False
        self.random_state_seed = 0
        self.use_n_xyz_xvyvzv_file = None
        self.use_barycenter_file = None
        self.size_distribution_steps = []
        self.start_at_distance_from_gmc = 0
        self.start_from_bin_file = None
        self.results_directory = ("Sim" + str(time_.localtime().tm_mday) +
                     "-" + str(time_.localtime().tm_mon) +
                     "-" + str(time_.localtime().tm_year) +
                     "_" + str(time_.localtime().tm_hour) +
                     ":" + str(time_.localtime().tm_min))
        assert os.path.exists(f)
        self.sim_file = f
        file_ext = f.split('.')
        file_ext = file_ext[len(file_ext) - 2] + "." + file_ext[len(file_ext) - 1]
        assert file_ext.lower() == 'oort.dat'
        reading_condensation_data = False
        reading_velocity_data = False
        reading_size_distribution_data = False
        with open(f, 'r') as file:
            for line in file.readlines():
                line = line.strip(" \n\r")
                if "#" in line:
                    pass  # comment line, ignore. Note comment lines can have # anywhere in the line.
                elif "<begin" in line:
                    # sets a flag so that we know we are in a data block.
                    print("entering data block")
                    if "gmc_condensation_data" in line:
                        reading_condensation_data = True
                    elif "velocity_data" in line:
                        reading_velocity_data = True
                    elif "size_distribution_data" in line:
                        reading_size_distribution_data = True
                elif "<end" in line:
                    print("exiting data block")
                    reading_condensation_data = False
                    reading_velocity_data = False
                    reading_size_distribution_data = False
                else:
                    if reading_condensation_data:
                        con = line.split(" ")
                        assert len(con) == 5
                        self.condensations.append(con)
                    elif reading_velocity_data:
                        v = line.split(" ")
                        assert len(v) == 4
                        if v[0] == "sun":
                            self.velocity_of_sun = np.array([float(v[1]), float(v[2]), float(v[3])])
                        elif v[0] == "gmc":
                            self.velocity_of_gmc = np.array([float(v[1]), float(v[2]), float(v[3])])
                    elif reading_size_distribution_data:
                        s = line.split(" ")
                        assert len(s) == 3
                        self.size_distribution_steps.append(s)
                    else:
                        setting, value = line.split("=")
                        assert setting != ""
                        assert value != ""
                        self.set_value(setting, value)

    def str_bool(self, s):
        if s == "False":
            return False
        elif s == "True":
            return True
        else:
            print("warning, string", s, "cannot be cast to boolean")
            return None

    def set_value(self, setting, value):
        """ Takes setting values from the oort.dat file and applies them.

        Units are as follows:

            distances: au, km for sizes
            times: earth years
            masses: kg
            densities: kg/m^3

        Currently accepted settings - anything else will be ignored:

            cloud_inner_radius
            cloud_outer_radius
            population
            percent_asteroids
            simulation_length
            minimum_comet_size#
            maximum_comet_size#
            comet_mass_density
            asteroid_mass_density
            results_directory
            integrator_timestep
            number_of_datapoints
            enable_drag
            enable_plummer_gravity
            enable_galactic_tide
            percent_asteroid (must be between 0 and 1)
            size_distribution : Size distribution type to use, one of: "meech-hainaut-marsden", others maybe later :)
            integrator : one of these https://rebound.readthedocs.io/en/latest/python_api.html#rebound.Simulation.integrator
            solar_system_file : one of "jupiter", "outer_planets", "all_planets" or "sun"

            # currently this sets the min/max size for both comets and asteroids.
            *Doesn't do anything but setup to accept these parameters so that they can do something in future.


        Args:
            setting: setting is apply.
            value: value to set it to.
        """
        with Switch(setting) as case:
            if case("cloud_inner_radius"):
                self.cloud_inner_radius = float(value)
            if case("cloud_outer_radius"):
                self.cloud_outer_radius = float(value)
            if case("population"):
                self.population = int(value)
            if case("simulation_length"):
                self.simulation_length = int(value) # This could be a float but a 100000.5 year simulation is a bit odd
            if case("minimum_comet_size"):
                self.minimum_comet_size = float(value)
            if case("minimum_asteroid_size"):
                self.minimum_asteroid_size = float(value)
            if case("maximum_comet_size"):
                self.maximum_comet_size = float(value)
            if case("maximum_asteroid_size"):
                self.maximum_asteroid_size = float(value)
            if case("comet_mass_density"):
                self.comet_mass_density = float(value)
            if case("asteroid_mass_density"):
                self.asteroid_mass_density = float(value)
            if case("results_directory"):
                self.results_directory = str(value)
            if case("number_of_orbit_plots"):
                self.number_of_orbit_plots = int(value)
            if case("create_gif_from_orbit_plots"):
                self.create_gif_from_orbit_plots = self.str_bool(value)
            if case("integrator_timestep"):
                self.integrator_timestep = float(value)
            if case("number_of_datapoints"):
                self.number_of_datapoints = int(value)
            if case("integrator"):
                self.integrator = str(value)
            if case("size_distribution"):
                self.size_distribution = str(value)
            if case("solar_system_file"):
                self.solar_system_file = os.path.join("solar_systems", str(value)+".bin")
            if case("percent_asteroid"):
                self.percent_asteroid = float(value)
            if case("enable_drag"):
                self.enable_drag = self.str_bool(value)
            if case("enable_plummer_gravity"):
                self.enable_plummer_gravity = self.str_bool(value)
            if case("enable_galactic_tide"):
                self.enable_galactic_tide = self.str_bool(value)
            if case("set_timestep_by_crossing_time"):
                self.set_timestep_by_crossing_time = float(value)
            if case("ism_density"):
                self.ism_density = float(value)
            if case("gmc_peak_density"):
                self.gmc_peak_density = float(value)
            if case("gmc_velocity_dispersion"):
                self.gmc_velocity_dispersion = float(value)
            if case("use_crossing_time_as_simulation_length"):
                self.use_crossing_time_as_simulation_length = self.str_bool(value)
            if case("solar_system_r_0_y"):
                self.solar_system_r_0_y = float(value)
            if case("create_path_plot"):
                self.create_path_plot = self.str_bool(value)
            if case("random_state_seed"):
                self.random_state_seed = int(value)
            if case("use_n_xyz_xvyvzv_file"):
                self.use_n_xyz_xvyvzv_file = str(value)
            if case("use_barycenter_file"):
                self.use_barycenter_file = str(value)
            if case("start_at_distance_from_gmc"):
                self.start_at_distance_from_gmc = float(value)
            if case("start_from_bin_file"):
                self.start_from_bin_file = str(value)

    def start_simulation(self):
        """Runs the simulation based on the given parameters.
        """
        # create the results directory
        path_number = 0
        while os.path.exists(self.results_directory):  # This only occurs if multiple runs occur in the same minute.
            path_number += 1
            self.results_directory += "_" + str(path_number)
        os.mkdir(self.results_directory)
        os.mkdir(os.path.join(self.results_directory, "particles"))  # for large simulations, having the particle data
        # mixed in with plots and analysis is very annoying
        # setup the molecular cloud
        sim.molecular_cloud = external.Molecular_cloud(start_at_distance_to_gmc=self.start_at_distance_from_gmc)
        for c in self.condensations:
            sim.molecular_cloud.add_condensation(float(c[0]), float(c[1]), float(c[2]), int(c[3]), float(c[4]))
        sim.molecular_cloud.set_velocities(self.velocity_of_sun, self.velocity_of_gmc)
        sim.molecular_cloud.set_molecular_cloud_parameters(self.ism_density, self.gmc_peak_density, self.gmc_velocity_dispersion)
        sim.molecular_cloud.set_r_0(self.solar_system_r_0_y)
        radius = sim.molecular_cloud.find_radius_of_cloud(0.05)
        # Sets up the crossing - calculates crossing time and plots a path.
        u = data.Unit_Conversion()
        crossing = u.time_SI_to_yr(sim.molecular_cloud.get_crossing_time(radius)) + u.time_SI_to_yr(sim.molecular_cloud.get_approach_time())
        if self.create_path_plot:
            sim.molecular_cloud.plot_path(int(crossing), z=0, scale=50, save_as=os.path.join(self.results_directory,
                                                                                        "path.png"))
        if self.use_crossing_time_as_simulation_length:
            self.simulation_length = crossing
            print("Setting simulation length to crossing time of", crossing, " years")
        if self.set_timestep_by_crossing_time > 0:
            self.integrator_timestep = crossing/self.set_timestep_by_crossing_time
            print("setting timestep to 1/"+str(self.set_timestep_by_crossing_time)+" of crossing time, which is",
                  self.integrator_timestep, "years")
        # setup the simulation object
        s = sim.Sim(self.simulation_length, solar_system=self.solar_system_file)
        if self.start_from_bin_file is None:
            s.set_random_state_seed(self.random_state_seed)
            s.set_densities(self.comet_mass_density, self.asteroid_mass_density, self.percent_asteroid)
            s.enable_external_forces(self.enable_drag, self.enable_plummer_gravity, self.enable_galactic_tide)
            # Create size distribution object
            dist = data.Size_Distribution(default_q=self.size_distribution)
            for i in range(len(self.size_distribution_steps)):
                dist.add_q(float(self.size_distribution_steps[i][2]), float(self.size_distribution_steps[i][0]),
                                                        float(self.size_distribution_steps[i][1]))
            # Generate comet sample
            if self.use_n_xyz_xvyvzv_file is None:
                s.generate_comet_sample(self.population,
                                    self.minimum_comet_size, self.maximum_comet_size,
                                    self.cloud_inner_radius, self.cloud_outer_radius,  # a
                                    0, 0.2,  # e
                                    0, np.pi,  # i
                                    0, 2 * np.pi,  # omega (angle from ascending node to x hat) 'argument of pericentre'
                                    0, 2 * np.pi,  # OMEGA 'Longitude of ascending node' (Reference line to ascending node)
                                    0, 1.0, dist)  # tau, time of pericentre - see note above
            else:
                print("importing data")
                s.import_n_xyz_xvyvzv_sample(self.use_n_xyz_xvyvzv_file, self.use_barycenter_file, self.population,
                                             self.minimum_comet_size,
                                             self.maximum_comet_size, dist, self.cloud_inner_radius)
            # Begin the simulation
            print("Beginning Integration of "+str(self.population)+
                  "particles for "+str(self.simulation_length)+" years.")
        else:  # If we are loading a binary, sizes, orbits, etc are already setup.
            s.load_binary(self.start_from_bin_file)
        results = s.integrate_simulation(integration_step=self.integrator_timestep,
                                         number_of_datapoints=self.number_of_datapoints,
                                         integrator=self.integrator)
        s.save_current_simulation_state(os.path.join(self.results_directory, "simulation.bin"))
        # save the results immediately to protect against a crash when plotting.
        print("Finished Integration")
        f = open(os.path.join(self.results_directory, "particles", "particle_list.txt"), "w")
        data.pack_into_file("eof", "goodbye", eof=True)  # this forces the packing method to write out everything in buffer.
        for i in range(len(results)):
            results[i].save_csv(os.path.join(self.results_directory, "particles"))
            f.write(str(results[i].get_label())+"\n")
        f.close()
        shutil.copy(self.sim_file, os.path.join(self.results_directory, "simulation_properties.txt"))
        print("Saved integration data in " + self.results_directory)
        # We need to tell Plotter where we saved the data otherwise it will look in the .tmp directory
        '''p = data.Plotter(results, location=self.results_directory)
        print("Generating aei Plot")
        p.plot_aei(save_as=os.path.join(self.results_directory, "aei_plot.png"), show=False, add_particle_labels=False)
        plots = []
        if self.number_of_orbit_plots != 0:
            print("Generating orbital plots, this can take some time.")
            for i in range(0, self.simulation_length, int(self.simulation_length/self.number_of_orbit_plots)):
                plots.append(p.plot_orbit(i, self.cloud_outer_radius))
            p.save_as_gif(plots, os.path.join(self.results_directory, "orbit_plot_animation"),
                            make_gif=self.create_gif_from_orbit_plots, move_raw=True)
        #shutil.rmtree(".tmp")'''
        print("Complete, exiting.")

def generate_ejection_stats(no_gmc, dynamics, drag, start_index, save_as):
    """This generate a csv of all the ejection by bin"""
    cleaned_dynamics, removed1 = process_multi(dynamics, no_gmc)
    print(len(removed1))
    # print(cleaned_dynamics)
    cleaned_dynamics_with_drag, removed2 = process_multi(drag, no_gmc)
    print(len(removed2))
    cleaned_drag_only, removed3 = process_multi(drag, dynamics, ex_particle_list=cleaned_dynamics_with_drag)
    print(len(removed3))
    bins = np.linspace(0.1, 2.0, 20)
    total_all = [0, 0, 0]
    ejected_all = [0, 0, 0]  # This is for total particles, regardless of bin
    for i, dataset in enumerate([[cleaned_dynamics, dynamics], [cleaned_dynamics_with_drag, drag], [cleaned_drag_only, drag]]):
        totals = np.zeros(len(bins)-1)
        ejections = np.zeros(len(bins)-1)
        data.unpack_header = None
        data.unpacked_data = None
        for j in range(len(dataset[0])):
            if j % 500 == 0:
                print(100*j/len(dataset[0]))
            p = data.Particle(label=dataset[0][j], location=dataset[1])
            if not p.get_ejected(start_index):
                total_all[i] += 1
                if p.get_ejected(-1):
                    ejected_all[i] += 1
                for k in range(1, len(bins)):
                    if bins[k-1] <= p.get_radius() < bins[k]:
                        totals[k-1] += 1
                        if p.get_ejected(-1):
                            ejections[k-1] += 1
                        break  # we found the right bin, no need to keep iterating.
            p.clear()

        if i == 0:
            f = open(save_as + "_dynamics_only.csv", "w")
        elif i == 1:
            f = open(save_as + "_dynamics_and_drag.csv", "w")
        else:
            f = open(save_as + "_drag_only", "w")
        f.write("Bins:, ")
        for j in range(1, len(bins)):
            f.write("bin_[" + str(bins[j - 1]) + "_to_" + str(bins[j]) + "), ")
        f.write("\nTotals:, ")
        for j in range(len(totals)):
            f.write(str(totals[j]) + ", ")
        f.write("\nEjections:, ")
        for j in range(len(ejections)):
            f.write(str(ejections[j]) + ", ")
        f.close()
        f = open(save_as+"_total_percent_ejected.txt", "w")
        labels = ["Dynamics only", "dynamics and drag", "drag only"]
        for i in range(len(total_all)):
            if total_all[i] > 0:
                f.write(labels[i]+": "+str(ejected_all[i])+"/"+str(total_all[i])+" "+str(ejected_all[i]/total_all[i])+"\n")
            else:
                f.write(labels[i] + ": " + str(ejected_all[i]) + "/" + str(total_all[i]) + " n/a\n")
        f.close()


def process_multi(ex, ctrl, ex_particle_list=None):
    """ The idea here is to find common particles between the experiment (ex) sim and the control. This returns a list
    of all particle in the ex data set, which have a different outcome to the control. i.e. if a particle is ejected
    in both control and experiment it is not returned but if it is bound in the control, but ejected in ex it is returned.
    particles bound in both are also returned
    """
    data.unpack_header = None
    data.unpacked_data = None
    particles_ex = []
    particles_ctrl = []
    exclude = []
    ctrl_initial = []
    ctrl_ejected = []
    # load the control data set
    if os.path.exists(ctrl + "/exclude_list.txt"):
        with open(ctrl + "/exclude_list.txt", "r") as exclude_file:
            for exclude_line in exclude_file:
                if not "#" in exclude_line:  # ignore comments about why a particle is excluded
                    exclude.append(exclude_line.strip("\n"))
    with open(ctrl + "/particle_list.txt", "r") as file_:
        for line__ in file_.readlines():
            part = line__.strip("\n")
            if part not in exclude:
                particles_ctrl.append(data.Particle(label=part, location=ctrl))
    # here we get all the info we need from the control because we can only open 1 data set at a time
    for i, part_ctrl in enumerate(particles_ctrl):
        ctrl_initial.append([part_ctrl.get_a()[0], part_ctrl.get_e()[0], part_ctrl.get_inc()[0]])
        ctrl_ejected.append(part_ctrl.get_e()[len(part_ctrl.get_e())-1] >= 1)
        part_ctrl.clear()
    # Load the experiment data set
    data.unpack_header = None
    data.unpacked_data = None
    if ex_particle_list is None:
        if os.path.exists(ex + "/exclude_list.txt"):
            with open(ex + "/exclude_list.txt", "r") as exclude_file:
                for exclude_line in exclude_file:
                    if not "#" in exclude_line:  # ignore comments about why a particle is excluded
                        exclude.append(exclude_line.strip("\n"))
        with open(ex + "/particle_list.txt", "r") as file_:
            for line__ in file_.readlines():
                part = line__.strip("\n")
                if part not in exclude:
                    particles_ex.append(data.Particle(label=part, location=ex))
    else:
        for j in range(len(ex_particle_list)):
            particles_ex.append(data.Particle(label=ex_particle_list[j], location=ex))
    removed_list = []
    particle_list = []
    for i, part_ex in enumerate(particles_ex):
        for j in range(0, len(ctrl_initial)):
            if part_ex.get_a()[0] == ctrl_initial[j][0] and part_ex.get_e()[0] == ctrl_initial[j][1] and part_ex.get_inc()[0] == ctrl_initial[j][2]:
                # we found a match, check ejection
                if ctrl_ejected[j] and part_ex.get_e()[len(part_ex.get_e())-1] >= 1:
                    removed_list.append(str(part_ex.get_label()))
                else:
                    particle_list.append(str(part_ex.get_label()))
                break
        part_ex.clear()
    return particle_list, removed_list

def plot_request(file):
    f = open(file)
    lines = f.read().split("\n")
    f.close()
    plotter = None
    for line_ in lines:
        li = line_.split()
        if len(li) > 0 and li[0] != '#':
            if li[0] == "load":
                data.unpack_header = None
                data.unpacked_data = None
                particles_ = []
                exclude = []
                if os.path.exists(li[1]+"/exclude_list.txt"):
                    with open(li[1]+"/exclude_list.txt", "r") as exclude_file:
                        for exclude_line in exclude_file:
                            if not "#" in exclude_line:  # ignore comments about why a particle is excluded
                                exclude.append(exclude_line.strip("\n"))
                with open(li[1] + "/particle_list.txt", "r") as file_:
                    for line__ in file_.readlines():
                        part = line__.strip("\n")
                        if part not in exclude:
                            particles_.append(part)
                plotter = data.Plotter(particles_, location=li[1])
            elif li[0] == "loadmulti":
                if len(li) == 3:
                    particles_, removed = process_multi(li[1], li[2])
                    print(str(len(removed))+" particle had same outcomes as control")
                else:
                    particles1, removed1 = process_multi(li[1], li[2])
                    print(str(len(removed1))+" particles had same outcome as control 1")
                    print(len(particles1))
                    particles_, removed2 = process_multi(li[1], li[3], ex_particle_list=particles1)
                    print(str(len(removed1))+" particles had same outcome as control 2")
                    print(len(particles_))
                plotter = data.Plotter(particles_, location=li[1])
            elif li[0] == "ejection_stats":
                no_gmc_control = "/media/wsa42/SIM_DATA/no_gmc_control/particles"
                dynamics_only = "/media/wsa42/SIM_DATA/small_centre_control/particles"
                with_drag = "/media/wsa42/SIM_DATA/small_centre/particles"
                start_index = 0
                if len(li) > 2:
                    for j in range(2, len(li)):
                        if li[j].split("=")[0] == "no_gmc_control":
                            no_gmc_control = li[j].split("=")[1]
                        if li[j].split("=")[0] == "dynamics_only":
                            dynamics_only = li[j].split("=")[1]
                        if li[j].split("=")[0] == "with_drag":
                            with_drag = li[j].split("=")[1]
                        if li[j].split("=")[0] == "start_index":
                            start_index = int(li[j].split("=")[1])
                generate_ejection_stats(no_gmc_control, dynamics_only, with_drag, start_index, li[1])
            elif plotter is not None:
                if li[0] == "aei":
                    subsample = 1
                    delta_a_thres = 0
                    alim = None
                    log = False
                    start_index = 0
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "subsample":
                                subsample = int(li[j].split("=")[1])
                            if li[j].split("=")[0] == "delta_a":
                                delta_a_thres = float(li[j].split("=")[1])
                            if li[j].split("=")[0] == "alim":
                                alim = int(li[j].split("=")[1])
                            if "use_log" in li[j]:
                                log = True
                            if li[j].split("=")[0] == "start_index":
                                start_index = int(li[j].split("=")[1])
                    plotter.plot_aei(add_particle_labels=False, subsample=subsample, delta_a_threshold=delta_a_thres, show=False, alim=alim, save_as=li[1], log_scale=log, start_index=start_index)
                elif li[0] == "sizedist":
                    ind1 = 0
                    ind2 = 0
                    ejected_ylim = 3000
                    remaining_ylim = 50000
                    ejected_max_bin = 2
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "initial_index":
                                ind1 = int(li[j].split("=")[1])
                            if li[j].split("=")[0] == "final_index":
                                ind2 = int(li[j].split("=")[1])
                            if li[j].split("=")[0] == "remaining_ylim":
                                remaining_ylim = int(li[j].split("=")[1])
                            if li[j].split("=")[0] == "ejected_ylim":
                                ejected_ylim = int(li[j].split("=")[1])
                            if li[j].split("=")[0] == "ejected_max_bin":
                                ejected_max_bin = float(li[j].split("=")[1])
                    plotter.plot_size_distribution(ind1, ind2, save_as=li[1], remaining_ylim=remaining_ylim, ejected_ylim=ejected_ylim, ejected_max_bin=ejected_max_bin)
                elif li[0] == "aei_dots":
                    ind = 0
                    alim = None
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "index":
                                ind = int(li[j].split("=")[1])
                            if li[j].split("=")[0] == "alim":
                                alim = int(li[j].split("=")[1])
                    plotter.plot_aei_dots(ind, save_as=li[1], alim=alim)
                elif li[0] == "find_ejected":
                    ind = 0
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "index":
                                ind = int(li[j].split("=")[1])
                    plotter.save_ejected_particle_list(ind, li[1])
                elif li[0] == "ejection_times":
                    start_index = 0
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "start_index":
                                start_index = int(li[j].split("=")[1])
                    plotter.plot_ejection_times(save_as=li[1], start_index=start_index)
                elif li[0] == "ejection_by_a":
                    start_index = 0
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "start_index":
                                start_index = int(li[j].split("=")[1])
                    plotter.plot_ejections_by_a(save_as=li[1], start_index=start_index)
                elif li[0] == "find_sizedist":
                    index = 0
                    if len(li) > 2:
                        for j in range(2, len(li)):
                            if li[j].split("=")[0] == "index":
                                index = int(li[j].split("=")[1])
                    plotter.fit_size_dist(save_as=li[1], index=index)
            else:
                print("error: plotter called before loading data")



# Check if a oort.dat file is provided
if len(sys.argv) > 1:
    if ".oort.dat" in sys.argv[1]:
        o = Oort_data(sys.argv[1])
        o.start_simulation()
    elif ".plot" in sys.argv[1]:
        plot_request(sys.argv[1])
else:
    # Go into command loop
    print("Welcome to the solar system GMC crossing simulator.\n"
          "This is version: " + version + "\n"
          "For any questions email wsa42@uclive.ac.nz\n")
    gmc = None
    sizedist = None
    while not end:
        uin = input(">").split()
        try:
            #print("Not enough parameters provided.")
            if uin[0] == "quit" or uin[0] == "q" or uin[0] == "exit":
                print("haere rā")
                end = True
            elif uin[0] == "simulate":
                print("You cannot do that at the prompt. Create an oort.dat file and pass that to this program through "
                      "the command line instead.")
            elif uin[0] == "generate":
                if uin[1] == "gmc":
                    params = input("Enter space separated: ism_density peak_density velocity_dispersion "
                                   "encounter_velocity >").split()
                    gmc = external.Molecular_cloud()
                    gmc.set_velocities(np.array([float(params[3]), 0, 0]), np.zeros(3))
                    gmc.set_molecular_cloud_parameters(float(params[0]), float(params[1]), float(params[2]))
                    print("generated gmc, its radius is ", str(gmc.find_radius_of_cloud(0.05)), "crossing time is",
                          str(gmc.get_crossing_time(gmc.find_radius_of_cloud(0.05))), "s")
                elif uin[1] == "sizedist":
                    params = input("enter the following: default q1 q1min q1max q1 q2min q2max ... qn qnmin "
                                   "qnmax>").split()
                    sizedist = data.Size_Distribution(default_q=float(params[0]), k=100)
                    for i in range(1, len(params), 3):
                        sizedist.add_q(float(params[i]), float(params[i+1]), float(params[i+2]))
                    print("Success")
            elif uin[0] == "plot":
                if uin[1] == "gmc" and uin[2] == "density":
                    if gmc is not None:
                        gmc.plot_number_density(z=0, scale=6)
                    else:
                        print("gmc not found, please use the generate gmc command first")
                elif uin[1] == "sim" or uin[1] == "simulation":
                    dir = uin[2]
                    print("Attempting to load data from "+dir)
                    try:
                        particles = []
                        with open(dir + "/particle_list.txt",
                                  "r") as file:
                            for line in file.readlines():
                                particles.append(line.strip("\n"))
                        p = data.Plotter(particles, location=dir)
                        print("Success!")
                        t = input("plot type and options? \n [aei opts / sizedist opts / orbits opts / a-e-changes "
                                  "opts \n/ leave blank for all plot types with default options]>").split()
                        if len(t) >= 1:
                            if t[0] == "aei":
                                if len(t) > 1:
                                    p.plot_aei(add_particle_labels=False, subsample=int(t[1]), delta_a_threshold=float(t[2]), show=False, save_as=t[3])
                                else:
                                    p.plot_aei(add_particle_labels=False, subsample=10, delta_a_threshold=0)
                            elif t[0] == "sizedist":
                                p.plot_size_distribution(int(t[1]), int(t[2]), show=False, save_as=t[3])
                            elif t[0] == "orbits":
                                p.plot_orbit(t[1], t[2], show=False, save_as=t[3])
                            elif t[0] == "a-e-changes":
                                p.e_a_changes_hist(t[1], t[2], show=False, save_as=t[3])
                        else:
                            p.plot_aei(add_particle_labels=False, subsample=10, delta_a_threshold=0)
                            p.plot_size_distribution(0, 499)
                            p.plot_orbit(499, 10000, show=True)
                            p.e_a_changes_hist(0, 499)
                    except FileNotFoundError:
                        print("Couldn't load data, is the file path correct?")
                elif uin[1] == "sizedist":
                    if sizedist is not None:
                        sizedist.plot_size_distribution()
                    else:
                        print("No sizedist object found. You can generate one with 'generate sizedist', or fit one to data with 'find sizedist'")
            elif uin[0] == "find":
                if uin[1] == "sizedist":
                    dir = uin[2]
                    print("Attempting to load data from " + dir)
                    try:
                        particles = []
                        with open(dir + "/particle_list.txt",
                                  "r") as file:
                            for line in file.readlines():
                                particles.append(data.Particle(label=line.strip("\n"), location=dir))
                        print("Success!")
                        index = input("I will now remove ejected particles from the dataset, what index should I look "
                                   "at?>")
                        index = int(index)
                        radii = []
                        for particle in particles:
                            if particle.get_e()[index] < 1:
                                radii.append(particle.get_radius())
                        print("Number of remain particles is "+str(len(radii)))
                        print(radii)
                        sizedist = data.Size_Distribution()
                        sizedist.fit_size_distribution(radii)
                    except FileNotFoundError:
                        print("Failed to read file, check file path is correct.")
                elif uin[1] == "gmc" and uin[2] == "mass":
                    if gmc is not None:
                        print("Please give the point to integrate around, and the radius of integration.")
                        params = input("Enter space separated: x y z radius>").split()
                        print("Starting integration, this can take a while.")
                        M = gmc.get_molecular_cloud_mass(float(params[0]), float(params[1]), float(params[2]), float(params[3]))
                        print("Integration complete: M="+str(M))
                    else:
                        print("gmc not found, please use the generate gmc command first")
            elif uin[0] == "h" or uin[0] == "help":
                help_cmd()
            else:
                print("Invalid argument(s)")
        except IndexError as e:
            print("list of arguments too short.")

