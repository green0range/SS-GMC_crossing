import unittest
import external
import numpy as np
import matplotlib.pyplot as plt

class TestParticle(unittest.TestCase):

    def setUp(self):
        self.mc = external.Molecular_cloud()

    def test_get_number_density(self):
        pass

    def test_get_drag_force(self):
        self.mc.set_velocities(np.array([1, 0, 0]), np.array([1, 0, 0]))
        # t = -1 is a special test value
        F = self.mc.get_drag_force(np.array([1, 0, 0]), 1, 0, 0, 0, -1)
        # the result is in the order of 1e-27, I have used atol to avoid floating point errors.
        np.testing.assert_allclose(F[0], -0.5*self.mc.get_density(1)*self.mc.get_drag_coefficient(), atol=1e-30)

    def test_high_density_stops_particles(self):
        self.mc.set_velocities(np.array([1, 0, 0]), np.array([1, 0, 0]))
        v = np.array([10., 0., 0.])
        a = np.array([0., 0., 0.])
        m = 1.
        N = 1533
        vels = np.zeros(N)
        for i in range(0, N):
            vels[i] = np.linalg.norm(v)
            v += a
            F = self.mc.get_drag_force(v, 20, 0, 0, 0, -2)
            a += F/m
        plt.plot(vels)
        plt.show()
        np.testing.assert_allclose(np.linalg.norm(v), 0, atol=0.1)

    def test_multiple_condensations(self):
        self.mc.add_condensation(-1, 0, 0, 2500, 2)
        self.mc.add_condensation(+1, 0, 0, 2500, 2)
        self.mc.r_0 = np.array([0, 0, 0])
        g = self.mc.plummer_sphere_g(0, 0, 0, 0)
        np.testing.assert_allclose(g, np.array([0., 0., 0.]), atol=1e-15)

    def test_plummer_sphere(self):
        self.mc = external.Molecular_cloud()  # clear existing
        self.mc.add_condensation(0, 0, 0, 10000, 1)
        g = self.mc.plummer_sphere_g(1, 1, 0, -1)
        np.testing.assert_allclose(g, np.array([-2.68243163e-10, -2.68243163e-10, -0.00000000e+00]), atol=1e-12)

    def test_lookup_table(self):
        mc = external.Molecular_cloud()
        mc.get_crossing_time(1)
        mc.set_r_0(0, override_radius=3)
        # pick 1000 random points and test if the lookup table result matches calculated result.
        n = 1000
        test_values = np.random.random_sample(size=(n, 3))
        for i in range(n):
            x = mc.lookup_number_density(test_values[i][0], test_values[i][1], test_values[i][2], 0, convert_units=False)
            y = mc.get_number_density(test_values[i][0], test_values[i][1], test_values[i][2], 0, convert_units=False)
            np.testing.assert_allclose(x, y, rtol=0.005)

