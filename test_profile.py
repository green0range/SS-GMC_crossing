import cProfile
import sim
import external
import numpy as np
import time

def profile_whole_program():
    s = sim.Sim(100)
    s.set_densities(2000, 1000, 0.1)
    sim.molecular_cloud = external.Molecular_cloud()
    sim.molecular_cloud.set_r_0(1)
    s.enable_external_forces(True, True, True)
    s.generate_comet_sample(1, 0.1, 100,  # num, radius
                                1e5, 2e5,  # a
                                0, 1,  # e
                                0, np.pi,  # i
                                0, 2 * np.pi,  # omega (angle from ascending node to x hat) 'argument of pericentre'
                                0, 2 * np.pi,  # OMEGA 'Longitude of ascending node' (Reference line to ascending node)
                                0, 1.0, "meech-hainaut-marsden")  # tau, time of pericentre - see note above

    profile = cProfile.run("s.integrate_simulation()", sort='cumulative')

def profile_number_density_methods():
    mc = external.Molecular_cloud()
    mc.get_crossing_time(5)
    mc.set_r_0(0, override_radius=3)
    t1 = time.time()
    mc.lookup_number_density(0, 0, 0, 0, convert_units=False)
    t2 = time.time()
    print("lookup generation time is", str(t2-t1), "s")
    n = 1000000
    test_values = np.random.random_sample(size=(n, 3))
    t1_lookup = time.time()
    for i in range(n):
        mc.lookup_number_density(test_values[i][0], test_values[i][1], test_values[i][2], 0, convert_units=False)
    t2_lookup = time.time()
    print("time for lookup table method is ", str(t2_lookup - t1_lookup), "s [without generation]")

    t1_calc = time.time()
    for i in range(n):
        mc.get_number_density(test_values[i][0], test_values[i][1], test_values[i][2], 0, convert_units=False)
    t2_calc = time.time()
    print("time for calculation method is ", str(t2_calc - t1_calc), "s")

profile_number_density_methods()