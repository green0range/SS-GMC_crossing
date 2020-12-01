"""This is for dumping temporary test code into
"""
import numpy as np
import sim
import data
import external
import matplotlib.pyplot as plt
import rebound
'''
particles = []
dir = "/home/wsa42/Desktop/sync/480/simulation_data/integrator_test/convergence2_ias15"
with open(dir+"/particle_list.txt",
          "r") as file:
    for line in file.readlines():
        particles.append(line.strip("\n"))

p = data.Plotter(particles, location=dir)

# p = data.Plotter("particle_in_1g_per_m3_neg_data_removed", location="/home/wsa42/Desktop/sync/480/Results/Testing/testing water density drag force")
'''

'''
s = sim.Sim(100)
s.set_densities(2000, 1000, 0.1)
sim.molecular_cloud = external.Molecular_cloud()
sim.molecular_cloud.set_r_0(2)
sim.molecular_cloud.add_condensation(4, 1, 0, 21520/2, 5)
sim.molecular_cloud.add_condensation(+0.5, 0, 0, 21520/2, 0.5)
sim.molecular_cloud.get_crossing_time(sim.molecular_cloud.find_radius_of_cloud(0.05))
sim.molecular_cloud.set_velocities(np.array([20000, 0, 0]), np.zeros(3))
s.enable_external_forces(True, True, True)
s.generate_comet_sample(100, 0.1, 100,  # num, radius
                            1e3, 1e5,  # a
                            0, 1,  # e
                            0, np.pi,  # i
                            0, 2 * np.pi,  # omega (angle from ascending node to x hat) 'argument of pericentre'
                            0, 2 * np.pi,  # OMEGA 'Longitude of ascending node' (Reference line to ascending node)
                            0, 1.0, "meech-hainaut-marsden")  # tau, time of pericentre - see note above

result = s.integrate_simulation(integrator="WHFAST")
p = data.Plotter(result, )
'''



#p.plot_aei_dots(0)
#p.plot_aei(add_particle_labels=False, subsample=20, delta_a_threshold=0)
#p.plot_size_distribution(0, 499)
#p.e_a_changes_hist(0, 499)
#p.eccentricity_stats(0, 499)
#p.plot_orbit(1000, 1.4e5, plot_backwards=200000, show=True)


#p.plot_orbit(9999, 1.4e5, plot_backwards=3000, show=True)

'''
n = 5000
sim.Sim.set_random_state_seed(43535)
e = sim.Sim.get_eccentricity_distribution(n)
a = []
for i in range(len(e)):
    a.append(sim.Sim.get_a_distribution(e[i], 1000, 100000, 40))
a = np.array(a)
print(len(e))
plt.scatter(a, e, marker='.', color="k", s=1)
plt.show()
'''
# Implementation of matplotlib function
from mpl_toolkits.axisartist.axislines import Subplot

fig = plt.figure()
ax = Subplot(fig, 111)
fig.add_subplot(ax)
fig.set_size_inches(12.8, 9.2)
ax.hist([1,2,1,2,2,2,3,4,6], bins=3)
ax.grid()
ax.set_xlabel("Time [yr]")
ax.set_ylabel("Number of particles ejected")

plt.show()