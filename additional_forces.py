import rebound
import numpy as np
import matplotlib.pyplot as plt
import time as time_
from matplotlib.ticker import StrMethodFormatter
import planets
import os
import csv
import data


sim = rebound.Simulation("planets.bin")
sim.status()
num_major_particles = len(sim.particles)  # a major particle is either a planet or the sun. (i.e. exclude comets)

number_of_orbit_plot_frames = 10
orbit_plotter = data.Plotter(None)

''' Generate a random sample of comets
'''
def generate_comet_sample(n, min_mass, max_mass,
                          min_a, max_a, min_e, max_e,
                          min_i, max_i, min_omega, max_omega,
                          min_Omega, max_Omega, min_tau, max_tau):  # note, tau should be between 0 and 1,
                                                                    # as a percent of period T.
    global sim
    comet_sample_mass = (max_mass - min_mass) * np.random.random_sample(n) + min_mass
    comet_sample_a = (max_a - min_a) * np.random.random_sample(n) + min_a
    comet_sample_e = (max_e - min_e) * np.random.random_sample(n) + min_e
    comet_sample_omega = (max_omega - min_omega) * np.random.random_sample(n) + min_omega
    comet_sample_Omega = (max_Omega - min_Omega) * np.random.random_sample(n) + min_Omega
    comet_sample_tau = (max_tau - min_tau) * np.random.random_sample(n) + min_tau
    comet_sample_inc = (max_i - min_i) * np.random.random_sample(n) + min_i

    for i in range(n):
        # calculate T
        T = np.sqrt((4*np.pi**2*comet_sample_a[i]**3)/(1))
        sim.add(m=comet_sample_mass[i],
                a=comet_sample_a[i],
                e=comet_sample_e[i],
                inc=comet_sample_inc[i],
                omega=comet_sample_omega[i],
                Omega=comet_sample_Omega[i],
                T=comet_sample_tau[i]*T)


generate_comet_sample(5, 1e-20, 1e-15,  # num, mass
                      25, 50,  # a
                      0, 1,  # e
                      0, np.pi,  # i
                      0, 2*np.pi,  # omega (angle from ascending node to x hat) 'argument of pericentre'
                      0, 2*np.pi,  # OMEGA 'Longitude of ascending node' (Reference line to ascending node)
                      0, 1)  # tau, time of pericentre - see note above
sim.move_to_com()  # Moves to the center of momentum frame

ps = sim.particles
year = 2.*np.pi
heartbeat_previous_t = 0

integration_time = 2000#00  # (earth) years of integration time

def drag(reb_sim):
    # using F_D = 1/2 rho v^2 C_D A
    rho = 1e-14
    C_D = 0.5
    A = 500 ** 2
    ps[1].ax -= (0.5 * rho * ps[1].vx ** 2 * C_D * A) / ps[1].m
    ps[1].ay -= (0.5 * rho * ps[1].vy ** 2 * C_D * A) / ps[1].m
    ps[1].az -= (0.5 * rho * ps[1].vz ** 2 * C_D * A) / ps[1].m


start_time = time_.time()


def are_you_OK(s):
    global heartbeat_previous_t, orbit_plotter
    t = s.contents.t/year
    if t > (heartbeat_previous_t + (integration_time/100)):
        percent_done = 100*t/integration_time
        runtime = time_.time() - start_time
        remaining_time = (100 - percent_done) * (runtime / percent_done)
        print("t:", str(t), "yrs;", str(percent_done)+"% complete; runtime: ", runtime, "s; estimated time remaining:", remaining_time, "s")
        heartbeat_previous_t = t

sim.heartbeat = are_you_OK

# sim.additional_forces = drag
# sim.force_is_velocity_dependent = 1

def record_initial_orbits(save_as, show_plot=True):
    initial_orbits_fig = rebound.OrbitPlot(sim, unitlabel="[AU]")
    plt.savefig(save_as)
    if show_plot:
        plt.show()

def integrate_simulation(samples_per_year=2):
    global sim
    Nout = integration_time * samples_per_year  # how many samples to take each year of integration?
    a = np.zeros([len(sim.particles) - num_major_particles, Nout])
    times = np.linspace(0., integration_time*year, Nout)
    e = np.zeros([len(sim.particles) - num_major_particles, Nout])
    inc = np.zeros([len(sim.particles) - num_major_particles, Nout])

    for i, time in enumerate(times):
        sim.integrate(time, exact_finish_time=0)
        for j in range(len(sim.particles) - num_major_particles):
            a[j][i] = ps[num_major_particles + j].a
            e[j][i] = ps[num_major_particles + j].e
            inc[j][i] = ps[num_major_particles + j].inc * (180 / np.pi)

        if i % number_of_orbit_plot_frames == 0:
            pass
        

    r = data.SimResult()
    r.save_a(a)
    r.save_e(e)
    r.save_inc(inc)
    r.save_t(times)
    return r


save_path = ("Sim"+str(time_.localtime().tm_mday) +
             "-"+str(time_.localtime().tm_mon) +
             "-"+str(time_.localtime().tm_year) +
             "_"+str(time_.localtime().tm_hour) +
             ":"+str(time_.localtime().tm_min))
path_number = 0
while os.path.exists(save_path):  # This only occurs if multiple runs occur in the same minute.
    path_number += 1
    save_path += "_"+str(path_number)
os.mkdir(save_path)
orbit_plotter.plot_orbits(sim, save_as=os.path.join(save_path, "initial_orbits_plot.png"))
result = integrate_simulation()
p = data.Plotter(result)
p.plot_aei(save_as=os.path.join(save_path, "aei_plot.svg"))
result.save_csv(os.path.join(save_path, "aeit.csv"))
