'''This file contains unit tests
'''
import sim
import unittest
from numpy import pi
import numpy as np
from scipy import optimize
import warnings



class TestSim(unittest.TestCase):
    def setUp(self):
        self.s = sim.Sim(1)

    def test_random_within_bounds(self):
        a = self.s.get_random_distribution(10, 1e-30, 1)
        assert len(a) == 10
        for i in range(len(a)):
            assert a[i] > 1e-30
            assert a[i] < 1

    def test_generate_comet_sample(self):
        self.assertRaises(AssertionError, lambda: self.s.generate_comet_sample(1,1, 0.5, 5, 10, 0, 0, 0, 0, 0, 0, 0,
                                                                               0, 0, 0, "meech-hainaut-marsden"))

    def test_are_you_okay(self):
        pass  # Not sure how to test

    def test_external_forces(self):
        pass  # external forces are not implemented

    def test_integrate_simulation(self):
        pass  # the only way I can think for how to test this is a regression test.

    def test_get_mass(self):
        np.testing.assert_allclose(self.s.get_mass(3, 0.25), 9*pi, atol=0.003, rtol=0)

    def test_get_size_distribution(self):
        '''
        The best way to test the distribution is to use Sim.plot_size_distribution, which makes a histogram of it.
        '''
        sizes = self.s.get_size_distribution(100, 0.1, 100, "meech-hainaut-marsden")
        assert len(sizes) == 100
        for i in range(100):
            assert sizes[i] >= 0.1 and sizes[i] <= 100

    def test_file_batch_system(self):
        """This test creates a batch at the batch size, which results
        in 1 full batch, at 1 empty batch, which is the most complicated
        case. It also tests when file write out dt == the integrator's internal dt.
        The `dry run` flag exists so this test can run without creating 1000 files.
        which would excessive for a unit test. The dry run batch size is 5 """
        self.s.set_densities(1,1, 0.03)
        self.s.generate_comet_sample(5, 0.1, 100,0,1,0,1,0,1,0,1,0,1,0,1,"meech-hainaut-marsden")
        assert len(self.s.integrate_simulation(integration_step=1, number_of_datapoints=1, dry_run=True)) == 5

    @staticmethod
    def power_relation(x, alpha, k):
        return k*(x**alpha)

    @staticmethod
    def fit_model():
        c = sim.Sim.get_size_distribution(100000, 0.1, 100, "meech-hainaut-marsden")
        y = np.zeros(100)
        x = np.linspace(0.1, 100, 100)
        for i in range(100):
            for j in range(100):
                #  this puts c into 100 bins.
                if x[i] <= c[j] < x[i + 1]:
                    y[i] += 1
        # Due to the nature of power laws, sometime everything goes into the first bin, but if this happens it is
        # impossible to fit a model, so just try again with a new distribution.
        if y[0] > 97:
            warnings.warn("Entered recursion because y[0] = "+str(y[0]))
            return TestSim.fit_model()
        popt, pcov = optimize.curve_fit(TestSim.power_relation, x, y, p0=[1, 1])
        fitted_alpha = popt[0]
        return fitted_alpha

    def test_model(self):
        pass
        '''
        target_alpha = -1.45
        fits = []
        # Each fit varies significantly so average over 50 fits
        for i in range(50):
            fits.append(self.fit_model())
        np.testing.assert_allclose(np.mean(fits), target_alpha, atol=0.25)
        print(np.mean(fits))
        '''

    def test_import_pl_data(self):
        x, v, M = sim.read_in_barycenter_data("/home/wsa42/Documents/University/Canterbury/2020/sync/480/code"
                                       "/.data_for_testing/pl.test_data")
        assert x == [0.5, 0.5, 0.5]
        assert v == [1., 1., 1.]