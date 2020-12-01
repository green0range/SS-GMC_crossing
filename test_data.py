"""Unit tests for the data module
"""
import data
import uuid
import unittest
import os
import shutil
from hashlib import sha256
import numpy as np


class TestParticle(unittest.TestCase):

    def setUp(self):
        self.p = data.Particle(pack_files=False)

    def tearDown(self):
        # todo: we should check that the main program cleans up as well (current doesn't as of 09-04-2020)
        #shutil.rmtree(".test")
        #shutil.rmtree(".tmp")
        pass

    def test_label_uuid(self):
        assert type(self.p.get_label()) is uuid.UUID or str

    def test_packed_files_unpack_correctly(self):
        p2 = data.Particle(pack_files=True)
        p3 = data.Particle(pack_files=True)
        p4 = data.Particle(pack_files=False)
        p5 = data.Particle(pack_files=False)
        p2.open_stream('w')
        p3.open_stream('w')
        p4.open_stream('w')
        p5.open_stream('w')
        d = np.random.random(14)
        p2.save_next(d[0], d[1], d[2], d[3], d[4], d[5], d[12])
        p4.save_next(d[0], d[1], d[2], d[3], d[4], d[5], d[12])
        p3.save_next(d[6], d[7], d[8], d[9], d[10], d[11], d[13])
        p5.save_next(d[6], d[7], d[8], d[9], d[10], d[11], d[13])
        p2.close_stream()
        p3.close_stream()
        p4.close_stream()
        p5.close_stream()
        assert p2.get_a() == p4.get_a()
        assert p3.get_x() == p3.get_x()

    @staticmethod
    def test_label_provided():
        p2 = data.Particle(label="myparticle2")
        assert p2.get_label() == "myparticle2"

    def test_open_stream(self):
        self.assertRaises(NotImplementedError, lambda: self.p.open_stream('asdf'))

    def test_check_var_a(self):
        self.assertRaises(AssertionError, lambda: self.p.check_var('this should be a float',  'a'))
        self.assertRaises(AssertionError, lambda: self.p.check_var(1j, 'a'))
        self.assertRaises(AssertionError, lambda: self.p.check_var(-3, 'a'))

    def test_check_var_e(self):
        self.assertRaises(AssertionError, lambda: self.p.check_var('this should be a float',  'e'))
        self.assertRaises(AssertionError, lambda: self.p.check_var(1.6, 'e'))
        self.assertRaises(AssertionError, lambda: self.p.check_var(-3, 'e'))

    def test_check_var_i(self):
        self.assertRaises(AssertionError, lambda: self.p.check_var('this should be a float',  'i'))
        self.assertRaises(AssertionError, lambda: self.p.check_var(182, 'i'))
        self.assertRaises(AssertionError, lambda: self.p.check_var(-3, 'i'))

    def test_check_var_t(self):
        self.assertRaises(AssertionError, lambda: self.p.check_var(-3, 't'))

    """
    Test removed because if it crashes (which is should) it might not close the stream.
    # this is technically an integration test since save_next uses check_var to check its inputs.
    def test_save_next_bad_input1(self):
        self.p.open_stream()
        self.assertRaises(AssertionError, lambda: self.p.save_next('hello', -1j, 1, 6, 3, "this shouldn't be a string", 1))
        self.p.close_stream()
    """
    @staticmethod
    def file_as_bytes(file):
        with file:
            return file.read()

    def test_save_next(self):
        self.p.open_stream('w')
        # Possible future check, check that the x,y,z points can exist on the ellipse described by e, a, i, etc
        v = [5.0, 0.1, 60.0, 1.4, 80.0, -60.0, 0.0]
        self.p.save_next(v[0], v[1], v[2], v[3], v[4], v[5], v[6])
        self.p.close_stream()
        # Test this file's hash against the excepted
        # If file formats, header etc are changed an new correct hash will have to be generated for this test.
        file = open(os.path.join(".tmp", str(self.p.get_label()) + ".csv"), 'r')
        h = sha256(self.file_as_bytes(file).encode('utf-8')).hexdigest()
        assert h == "44755080a80e3302906de75f2a389a86c7915168ccc3b243fc68c1f36e7bf09e"

    # I don't think close_stream can be effective tested since it just closes a file

    def test_read_gets(self):
        # create a particle with some data to test. These commands have been tested above
        p2 = data.Particle()
        p2.open_stream('w')
        v = [5.0, 0.1, 60.0, 1.4, 80.0, -60.0, 0.0]
        p2.save_next(v[0], v[1], v[2], v[3], v[4], v[5], v[6])
        p2.close_stream()
        # check the values can be retrieved
        '''Note this also tests read_in(), read_in_t(), clear() and read_in_xyz() since these are
            helper methods for the gets. This blurs the line between unit testing and regression testing. 
        '''
        assert p2.get_a() == [v[0]]
        assert p2.get_e() == [v[1]]
        assert p2.get_inc() == [v[2]]
        assert p2.get_t() == [v[3]]
        assert p2.get_x() == [v[4]]
        assert p2.get_y() == [v[5]]
        assert p2.get_z() == [v[6]]

    def test_save_csv_no_dir(self):
        # create a particle with some data to test. These commands have been tested above
        p2 = data.Particle()
        p2.open_stream('w')
        v = [5.0, 0.1, 60.0, 1.4, 80.0, -60.0, 0.0]
        p2.save_next(v[0], v[1], v[2], v[3], v[4], v[5], v[6])
        p2.close_stream()
        # check for an error if we try to save somewhere nonexistent
        self.assertRaises(FileNotFoundError, lambda: p2.save_csv("some dir that does not exits"))

    def test_save_csv_same_hash(self):
        # create a particle with some data to test. These commands have been tested above
        p2 = data.Particle()
        p2.open_stream('w')
        v = [5.0, 0.1, 60.0, 1.4, 80.0, -60.0, 0.0]
        p2.save_next(v[0], v[1], v[2], v[3], v[4], v[5], v[6])
        p2.close_stream()
        # Check that the moved file has the same hash. i.e. the contents don't change when copying
        if not os.path.exists(".test"):
            os.mkdir(".test")
        p2.save_csv(".test")
        file = open(os.path.join(".test", str(p2.get_label()) + ".csv"), 'r')
        h = sha256(self.file_as_bytes(file).encode('utf-8')).hexdigest()
        assert h == "44755080a80e3302906de75f2a389a86c7915168ccc3b243fc68c1f36e7bf09e"


class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.p = data.Plotter('particle_bob', location='.data_for_testing')

    def test_plot_aei(self):
        print("Please check the plot presented looks visually OK")
        self.p.plot_aei(save_as=os.path.join(".test", "plot.png"))
        assert os.path.exists(os.path.join(".test", "plot.png"))

    def test_plot_orbits(self):
        orb_plt = self.p.plot_orbit(0, 60)
        assert type(orb_plt) is str
        assert os.path.exists(orb_plt)
        """Same goes with this, I can check that it returns a valid file path,
        but not that the file is 'correct'. """

    def test_save_as_gif(self):
        #self.p.plot_orbit
        pass

class TestUnitConverter(unittest.TestCase):
    def setUp(self):
        self.u = data.Unit_Conversion()

    def test_dist(self):
        assert self.u.dist_to_SI(1) == 1.495978707e11
        assert self.u.dist_from_SI(1.495978707e11) == 1
        assert self.u.dist_au_to_pc(1) == 4.84814e-6  # checking against Google's conversion
        assert self.u.dist_SI_to_pc(3.086e16) == 1
        assert self.u.dist_pc_to_SI(1) == 3.086e16

    def test_vel(self):
        pass  # todo: unit tests for all these

    def test_acc(self):
        pass

    def test_mass(self):
        pass

    def test_time(self):
        pass

