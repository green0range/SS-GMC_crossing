import numpy as np
import matplotlib.pyplot as plt
import rebound
import data
import scipy.integrate
from astropy import units as aunit
from astropy import constants as const
from matplotlib import ticker

class Molecular_cloud():

    def __init__(self, start_at_distance_to_gmc=0):
        # These velocities are placeholders
        self.v_sun = np.array([20000, 0, 0])  # assumes sun is travelling in x direction at 20km/s
        self.v_molecular_cloud = np.array([10000, 0, 0])  # assume MC is orbiting a half the speed of the sun.
        # The rocky particle mass needs a better estimate, currently I have taken it to be 50 silicon dioxide molecules
        # I am assuming molecular hydrogen, H_2.
        self.masses = {"H": 2*1.6735575e-27, "He": 6.6464764e-27, "rocky_particles": 50*4.6637066e-26+100*2.6566962e-27}
        self.mc_components = {"H": 0.9, "He": 0.099, "rocky_particles": 0.001}
        assert sum([self.mc_components["H"], self.mc_components["He"], self.mc_components["rocky_particles"]]) == 1
        self.drag_coefficient = 2.0
        self.u = data.Unit_Conversion()

        self.ism_density = 200
        self.peak_density = 5000
        self.velocity_dispersion = 840
        self.r_0 = np.array([-10, 0, 0])
        self.start_at_distance_to_gmc = start_at_distance_to_gmc

        # A plummer sphere is created at the centre of each condensation.
        self.condensations = [[0, 0, 0, 25000, 5]]

        self.crossing_time = None
        self._gmc_density_lookup_table = None

        self.set_r_0(0)

        '''
        All variables starting with "_" are pre-assigned for usage by methods which run frequently in order
        to avoid slow down issues associated with memory assignment.
        '''
        self._convert_mc_frame_vector = np.array([0., 0., 0.])
        self._lookup_resolution = 30
        self._get_mass_sp_centre = [0,0,0]

    def set_velocities(self, v_sun, v_mc):
        assert len(v_sun) == len(v_mc) == 3
        assert type(v_sun) == type(v_mc) == np.ndarray
        self.v_sun = v_sun
        self.v_molecular_cloud = v_mc

    def set_r_0(self, y, override_radius=0):
        '''If we set r_0 to be far away from the molecular cloud, then we waste computing time while it approaches,
        so here we pick a y coordinate and find where the x coordinate is that corresponds to that y coord at the edge
        of the cloud.'''
        if override_radius == 0:
            R = self.find_radius_of_cloud(0.05)
        else: R = override_radius
        assert R > y  # if we pick a y outside/above the GMC, then it wont work.
        x = -np.sqrt(R ** 2 - y ** 2) - self.start_at_distance_to_gmc  # x is negative because we assume travel from left to right
        self.r_0 = np.array([x, y, 0])

    def add_condensation(self, x, y, z, M, R):
        '''Adds a condensation to the gravitational model.

        Params:
            x, y, z: Position of condensation in pc relative to centre
            M: Mass of condensation
            R: radius of condensation (radius in which is a particle is bound to this condensation, not another one)
        '''
        # Delete the default condensation before adding new ones.
        if len(self.condensations) == 1 and np.array_equal(self.condensations[0][:3], np.zeros(3)):
            self.condensations = [[x, y, z, M, R]]
        else:
            self.condensations.append([x, y, z, M, R])

    def plummer_sphere_g(self, x, y, z, t):
        """Gives a plummer sphere gravitational field centred around the molecular cloud.

        """
        G = 6.6743e-11  # gravitational constant
        g_x = 0  # = np.array([0., 0., 0.])
        g_y = 0
        g_z = 0
        for i in range(len(self.condensations)):
            M = self.u.mass_to_SI(self.condensations[i][3])
            a = self.u.dist_pc_to_SI(self.condensations[i][4])

            """convert from the heliocentric frame to molecular cloud centric frame
            x, y, z are in pc.
            """
            r = self.convert_to_mc_frame(x, y, z, t)  # - np.array(self.condensations[i][:3])
            r_x = self.u.dist_pc_to_SI(r[0] - self.condensations[i][0])
            r_y = self.u.dist_pc_to_SI(r[1] - self.condensations[i][1])
            r_z = self.u.dist_pc_to_SI(r[2] - self.condensations[i][2])

            r_sqrd = r_x**2 + r_y**2 + r_z**2  #self.u.dist_pc_to_SI(np.linalg.norm(position))**2

            A = (-1*G*M)/((r_sqrd + (a**2))**(3/2))
            g_x += A*r_x
            g_y += A*r_y
            g_z += A*r_z
            #A = float(A.to(1/aunit.s**2) * aunit.s**2)
            #g += (A * self.u.dist_pc_to_SI(position))
                #np.array([A*self.u.dist_pc_to_SI(x), A*self.u.dist_pc_to_SI(y), A*self.u.dist_pc_to_SI(z)])
        return np.array([g_x, g_y, g_z])

    def galactic_tide(self, z, x, check_conversion=False):
        """Model of galactic tide. Only the z component is considered since it is dominates

        Note: this assumes the sun is perfectly in the plane of the galaxy and the only height above/below to consider
        is the height the comet is above/below the sun.

        Params:
            z: z coordinate in the heliocentric frame, in au.

        returns:
            acceleration due to the galactic tide.
        """
        G = 6.6743e-11  # m^3/(kg s^2)
        rho_0 = 0.15 * 6.76599e-20  # M_sun/pc^3  => kg/m^3
        z = z * 149597870700.0  # au => m
        z = -x*np.sin(1.05068821) + z*np.cos(1.05068821)  # the solar system is inclined to the galactic plane at
        # 60.2 degrees.
        A = 14.4 / 3.086e16  # km/(s kpc) => 1/s
        B = -12.0 / 3.086e16  # km/(s kpc) => 1/s
        Fz = -4*np.pi*G*rho_0*z + 2*(B**2 - A**2)*z
        if check_conversion:
            z = z * aunit.m
            rho_0 = 0.15 * aunit.solMass/(aunit.pc**3)   #solar_masses/pc^3
            A = 14.4 * aunit.km/(aunit.s*aunit.kpc)  # +/- 1.2, km/s/kpc
            B = -12.0 * aunit.km/(aunit.s*aunit.kpc)  # +/- 2.8, km/s/kpc
            G = const.G  # gravitational constant in pc*km^2/(M_sun*s^2)
            F_z = -4*np.pi*G*rho_0*z + 2*(B**2 - A**2)*z
            F_z_astropy = F_z.to(aunit.m/aunit.s**2) * aunit.s**2/aunit.m  # needs to return a unitless value
            np.testing.assert_allclose(Fz, F_z_astropy, atol=1e-15)
        return Fz


    def get_drag_coefficient(self):
        return self.drag_coefficient

    def convert_to_mc_frame(self, x, y, z, t):
        """Converts from a heliocentric reference frame to a molecular-cloud-centric reference frame
        Params:
            x: x position, in au, in heliocentric frame
            y: y position, in au, in heliocentric frame
            z: z position, in au, in heliocentric frame
            t: time in seconds since solar system's position was r_0

        Returns: position vector, in pc, in molecular-cloud-centric frame
        """
        if t == -1:  # for testing
            return np.array([1., 1., 0.])
        else:
            self._convert_mc_frame_vector[0] = self.u.dist_au_to_pc(x)
            self._convert_mc_frame_vector[1] = self.u.dist_au_to_pc(y)
            self._convert_mc_frame_vector[2] = self.u.dist_au_to_pc(z)
            #r = np.array([self.u.dist_au_to_pc(x), self.u.dist_au_to_pc(y), self.u.dist_au_to_pc(z)])
            return self.r_0 + self._convert_mc_frame_vector + (self.u.vel_SI_to_pcs(self.v_sun - self.v_molecular_cloud) * t)

    def get_mean_molecular_mass(self):
        """Returns the mean molecular mass, weighted by the chemical abundance of each molecule in the cloud."""
        return (self.mc_components["H"]*self.masses["H"] +
                                          self.mc_components["He"]*self.masses["He"] +
                                          self.mc_components["rocky_particles"]*self.masses["rocky_particles"])

    def get_crossing_time(self, R):
        """This finds the crossing time of the sun. It uses the fact that when the sun is entering and exiting
        the GMC, |r| = R, where R is the radius of the GMC and |r| is the magnitude of the sun's position vector.
        r = r_0 + v*t => a quadratic which can be solved for t. The Crossing time is the difference between the
        solutions for t.

        Assumptions:
            v is constant.

        Params:
            R: radius of the molecular cloud."""
        R = self.u.dist_pc_to_SI(R)
        v = self.v_sun - self.v_molecular_cloud
        a = np.linalg.norm(v)**2
        b = 2*(self.r_0[0]*v[0] + self.r_0[1]*v[1] + self.r_0[2]*v[2])
        c = np.linalg.norm(self.r_0)**2 - R**2
        t_1 = (-b**2 - np.sqrt(b**2 - 4*a*c))/(2*a)
        t_2 = (-b**2 + np.sqrt(b**2 - 4*a*c))/(2*a)
        crossing_time = t_2 - t_1
        # print("Crossing time is", str(crossing_time), "s")
        # print("Crossing time is", self.u.time_SI_to_yr(crossing_time), "years")
        self.crossing_time = crossing_time
        return crossing_time

    def get_approach_time(self):
        """ This time it takes to approach the gmc. Returns in seconds"""
        v = np.linalg.norm(self.v_sun - self.v_molecular_cloud)
        assert v != 0
        return self.u.dist_pc_to_SI(self.start_at_distance_to_gmc)/v

    def set_molecular_cloud_parameters(self, ism_density, peak_density, vel_dispersion):
        self.ism_density = ism_density
        self.peak_density = peak_density
        self.velocity_dispersion = vel_dispersion

    def lookup_number_density(self, x, y, z, t, convert_units=True):
        """ This method looks up the number density in a lookup table for faster processing times.
        To calculate the number density exactly, use `get_number_density`

        This should return the same value as get_number_density, within a resolution error.

        Note if the lookup table is not initialised, (i.e. on first run) this will generate the lookup table
        using get_number_density to do calculations.

        Do not use this for plots, as it only generates a lookup table for the solar systems path, not the entire GMC.

        :param x:
        :param y:
        :param z:
        :param t:
        :return:
        """
        if t == -1:
            return 1.  # most basic test case
        elif t == -2:
            return 999999999999999999.  # this is to check that a particle is stopped if density -> inf
        else:
            """x,y,z are given in au, with (0,0,0) being the at the sun. This needs to be converted to parsecs,
            and the coordinates shifted to the frame where (0,0,0) is the centre of the molecular cloud.

            The convert_units flag is so I can generate density plots directly in parsecs
            """
            #
            assert t >= 0
            if convert_units:
                position = self.convert_to_mc_frame(x, y, z, t)
                x = position[0]
                y = position[1]
                z = position[2]
            self._lookup_resolution = 30
            x1 = self.r_0[0]
            y1 = self.r_0[1]
            z1 = self.r_0[2]
            if self._gmc_density_lookup_table is None:
                """Find the rectangular "cylinder" that the solar system will pass through, create a lookup table for all 
                values within that cylinder. delta is the 'radius' to compute from the centeral line that the sun 
                will travel down. resolution is the number of points to store per parsec, points between are 
                calculated by weighted average of neighbouring points.
                """
                delta = 1
                x2 = self.r_0[0] + (self.u.vel_SI_to_pcs(self.v_sun - self.v_molecular_cloud) * self.crossing_time)[0]
                y2 = self.r_0[1] + (self.u.vel_SI_to_pcs(self.v_sun - self.v_molecular_cloud) * self.crossing_time)[1]
                z2 = self.r_0[2] + (self.u.vel_SI_to_pcs(self.v_sun - self.v_molecular_cloud) * self.crossing_time)[2]
                shape_x = int((x2 - x1) * self._lookup_resolution + 2. * delta * self._lookup_resolution)
                shape_y = int((y2 - y1) * self._lookup_resolution + 2. * delta * self._lookup_resolution)
                shape_z = int((z2 - z1) * self._lookup_resolution + 2. * delta * self._lookup_resolution)
                self._gmc_density_lookup_table = np.zeros(shape=(shape_x, shape_y, shape_z))
                print(np.shape(self._gmc_density_lookup_table))
                """Fill in the lookup table.
                This will be   s l o w,     but only needs to run once."""
                print("Generating density lookup table, this may take a while")
                for i in range(np.shape(self._gmc_density_lookup_table)[0]):
                    for j in range(np.shape(self._gmc_density_lookup_table)[1]):
                        for k in range(np.shape(self._gmc_density_lookup_table)[2]):
                            x_ = (i / self._lookup_resolution) + x1
                            y_ = (j / self._lookup_resolution) + y1
                            z_ = (k / self._lookup_resolution) + z1
                            self._gmc_density_lookup_table[i][j][k] = self.get_number_density(x_, y_, z_, 0, convert_units=False)
                print("Successfully generated density lookup table!")

            # To average between points we need some Trilinear Interpolation
            # https://en.wikipedia.org/wiki/Trilinear_interpolation
            i, j, k = (x - x1)*self._lookup_resolution, (y - y1)*self._lookup_resolution, (z - z1)*self._lookup_resolution
            x_d = i - int(i)
            y_d = j - int(j)
            z_d = k - int(k)
            # define the 8 corners of the cube
            try:
                c_000 = self._gmc_density_lookup_table[int(i)][int(j)][int(k)]
                c_001 = self._gmc_density_lookup_table[int(i)][int(j)][int(k+1)]
                c_101 = self._gmc_density_lookup_table[int(i+1)][int(j)][int(k+1)]
                c_100 = self._gmc_density_lookup_table[int(i+1)][int(j)][int(k)]
                c_010 = self._gmc_density_lookup_table[int(i)][int(j+1)][int(k)]
                c_110 = self._gmc_density_lookup_table[int(i+1)][int(j+1)][int(k)]
                c_011 = self._gmc_density_lookup_table[int(i)][int(j+1)][int(k+1)]
                c_111 = self._gmc_density_lookup_table[int(i+1)][int(j+1)][int(k+1)]
            except IndexError:
                # print("Warning: position outside lookup table, reverting to calculation")
                return self.get_number_density(x, y, z, t, convert_units=False)

            c_00 = c_000*(1-x_d) + c_100*x_d
            c_01 = c_001*(1-x_d) + c_101*x_d
            c_10 = c_010*(1-x_d) + c_110*x_d
            c_11 = c_011*(1-x_d) + c_111*x_d

            c_0 = c_00*(1-y_d) + c_10*y_d
            c_1 = c_01*(1-y_d) + c_11*y_d

            c = c_0*(1-z_d) + c_1*z_d
            return c

    def get_number_density(self, x, y, z, t, convert_units=True, use_cm=False):
        """
        Returns: Number density in particles per m^3

        The cases where t is -1 or -2 are for testing. t must not be negative unless it is these specific test
        values.
        """
        if t == -1:
            return 1.  # most basic test case
        elif t == -2:
            return 999999999999999999.  # this is to check that a particle is stopped if density -> inf
        else:
            """x,y,z are given in au, with (0,0,0) being the at the sun. This needs to be converted to parsecs,
            and the coordinates shifted to the frame where (0,0,0) is the centre of the molecular cloud.
            
            The convert_units flag is so I can generate density plots directly in parsecs
            """
            #
            assert t >= 0
            if convert_units:
                position = self.convert_to_mc_frame(x, y, z, t)
                x = position[0]
                y = position[1]
                z = position[2]

            """Set up the model's constants"""
            n_init_uniform = self.ism_density   # 200  # cm^-3
            n_init_peak = self.peak_density   #5000  # cm^-3
            velocity_dispersion = self.velocity_dispersion  # = 840  # m/s
            g = 1.3  # prolate spheroid aspect ratio
            A = 0.34  # Modulation
            G = 6.67430e-11  # gravitational constant in SI
            n_core = n_init_peak * 1e6  # core density in m^-3
            m = self.get_mean_molecular_mass()  # mean molecular mass

            # constants
            C = 50
            D = 48

            a = np.sqrt(velocity_dispersion**2/(4*np.pi*m*G*(n_init_peak - n_init_uniform)*1e6))
            a_sqrd = self.u.dist_SI_to_pc(a)**2
            beta_sqrd = ((x/g)**2+y**2)/a_sqrd
            gamma_sqrd = 10+beta_sqrd
            delta_sqrd = 12+beta_sqrd
            zeta_sqrd = z**2/a_sqrd

            '''This gives the initial spheroidal and uniform densities'''
            n_spheroidal = (n_init_peak - n_init_uniform)*((C/(gamma_sqrd+zeta_sqrd)) - (D/(delta_sqrd+zeta_sqrd)))

            '''Modulate the density to create repeating structures'''
            l_m = velocity_dispersion/np.sqrt(2*np.pi*G*m*n_core)  # modulation scale length, this is in meters
            l_m = self.u.dist_SI_to_pc(l_m)
            Z = 5
            Lambda_l = Z/(2*l_m)


            # Rotate modulation by some angle
            phi = np.pi/8
            modulation_direction = x*np.cos(phi) + y*np.sin(phi)
            # Also a circular modulation
            # modulation_direction = (x**2+y**2)/np.sqrt(2)

            alpha = A*np.cos(modulation_direction/l_m)
            F = ((1 - A**2)/(1 - alpha**2))*(1/(1+alpha*(1/np.cosh(Lambda_l))) - ((2*alpha*(np.cosh(Lambda_l)/np.sinh(Lambda_l)))/(np.sqrt(1-alpha**2)))*np.arctan(((1-alpha)/np.sqrt(1-alpha**2))*np.tanh(Lambda_l/2)))

            '''Return the sum of column density and the background spherical-uniform density
            The `use_cm` flag returns the number density in 1/cm^3, which looks nicer on graphs.
            Otherwise 1/m^3 is used.
            '''
            if use_cm:
                return (F*n_spheroidal + n_init_uniform)
            else:
                return (F*n_spheroidal + n_init_uniform)*1e6

    def get_density(self, number_density):
        return number_density*self.get_mean_molecular_mass()

    @staticmethod
    def integration_test(x, y, z):
        """This represents a cube of constant density of 1, with volume of 1pc^3.
        """
        if -0.5 < x < 0.5:
            if -0.5 < y < 0.5:
                if -0.5 < z < 0.5:
                    return 1
        return 0

    @staticmethod
    def integration_test_spherical(r, theta, phi):
        """This represents a sphere of unit volume and density"""
        if r <= (3/(4*np.pi))**(1/3):
            return 1
        return 0

    def get_number_density_in_spherical_polar(self, r, theta, phi):
        """ This is basically just a helper function for get_molecular_cloud_mass.
        It allows it to integrate over a radius around a point.
        Note: using the phi=azimuthal, theta=polar convention"""
        x = self._get_mass_sp_centre[0] + r*np.cos(phi)*np.sin(theta)
        y = self._get_mass_sp_centre[1] + r*np.sin(phi)*np.sin(theta)
        z = self._get_mass_sp_centre[2] + r*np.cos(theta)
        return r**2 * np.sin(theta) * self.get_number_density(x, y, z, 0, convert_units=False, use_cm=False)

    def get_molecular_cloud_mass_box(self, x1, x2, y1, y2, z1, z2):
        """This preforms a numerical integration over the molecular cloud's density to get its mass.

        This is very slow but gives 155431.65841166733 solar masses when integrated from -10 to 10 in all axes.

        Checks: in the central 5pc, M = 5372, just less than the expected 6100

        Params:
            i1: (i = x, y, z) Lower bound of the integral, should be a point outside the GMC
            i2: (i = x, y, z) Upper bound of the integral, should be outside the GMC but on the other side from i1

        Returns: Mass of molecular cloud in solar masses.

        Note the 3.086e16^3 from converting pc^3
        """
        N = scipy.integrate.tplquad(self.get_number_density, x1, x2, y1, y2, z1, z2, args=(0, False, False))
        #print("raw:" + str(N))
        return self.u.mass_from_SI(N[0]*self.get_mean_molecular_mass()*(3.086e16**3))

    def get_molecular_cloud_mass(self, x, y, z, r):
        """Integrates the number density spherically around a point.
        """
        self._get_mass_sp_centre = np.array([x, y, z])
        ranges = [(0, r), (0, np.pi), (0, 2*np.pi)]
        options = {"limit": 10, "epsabs": 0.001}
        N = scipy.integrate.nquad(self.get_number_density_in_spherical_polar, ranges, opts=options)
        #print("raw:" + str(N))
        return self.u.mass_from_SI(N[0]*self.get_mean_molecular_mass()*(3.1e16**3))

    def get_drag_force(self, comet_v, cross_section_area, x, y, z, t):
        """Gets the drag force on the comet
        Params:
            comet_v: numpy array of the comets velocity vector in SI units
            cross_section_area: the area of the comet in m^2
            x: x position in au
            y: y position in au
            z: z position in au

        returns: The force as a vector in Newtons"""
        assert type(comet_v) == np.ndarray
        v_sc = comet_v + self.v_sun
        v_rel = v_sc - self.v_molecular_cloud
        #print("comet vel: "+str(comet_v))
        density = self.get_density(self.lookup_number_density(x, y, z, t))
        force_mag = -0.5 * density * np.linalg.norm(v_rel)**2 * self.drag_coefficient * cross_section_area
        #print("F mag is "+str(force_mag))
        return force_mag * (v_rel/np.linalg.norm(v_rel))


    def plot_number_density(self, z=0, scale=50):
        """Makes a contour plot of the number density through a plane in the z axis.
        Params:
            z: the plane through which to slice
            scale: scale of the plot in parsecs. (will plot from x,y = -scale to +scale)"""
        assert scale > 0
        x = np.linspace(-1*scale, scale, 1000)
        y = np.linspace(-1*scale, scale, 1000)
        X, Y = np.meshgrid(x, y)
        n = self.get_number_density(X, Y, z, 0, convert_units=False, use_cm=True)
        #print(n)
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, n, 50, locator=ticker.LogLocator(base=10, subs=(1.0, 3.0, 6.0)))
        cbar = fig.colorbar(cp)  # Add a colorbar to a plot
        ax.set_title('Gas density at plane of z = '+str(z))
        ax.set_xlabel('x (pc)')
        ax.set_ylabel('y (pc)')
        cbar.set_label('number density ($cm^{-3})$', rotation=270, labelpad=30)
        plt.tight_layout()
        plt.show()

    def find_radius_of_cloud(self, percent_of_ism_to_consider_edge, show_plot=False):
        """Plots the density as a function of radius."""
        r = np.linspace(0, 50, 10000)
        n = self.get_number_density(r, 0, 0, 0, convert_units=False, use_cm=True)
        R = 0
        for i in range(len(r)):
            if abs(n[i] - self.ism_density) < percent_of_ism_to_consider_edge*self.ism_density:
                R = r[i]
                break
        if show_plot:
            plt.semilogy(r, n)
            plt.show()
        return R

    def plot_path(self, t_final, z=0, scale=500, save_as=None):
        assert scale > 0
        x_sol = []
        y_sol = []
        for i in range(0, t_final, int(t_final/10)):
            t = self.u.time_yr_to_SI(i)
            r = self.convert_to_mc_frame(0, 0, z, t)
            x_sol.append(r[0])
            y_sol.append(r[1])
        print(x_sol)

        x = np.linspace(-1 * scale, scale, 1000)
        y = np.linspace(-1 * scale, scale, 1000)
        X, Y = np.meshgrid(x, y)
        n = self.get_number_density(X, Y, z, 0, convert_units=False, use_cm=True)
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, n, 20)
        cbar = fig.colorbar(cp)  # Add a colorbar to a plot
        ax.plot(x_sol, y_sol, color="c", ls="--")
        con_x = []
        con_y = []
        con_m = []
        for i in range(len(self.condensations)):
            con_x.append(self.condensations[i][0])
            con_y.append(self.condensations[i][1])
            if self.condensations[i][2] == 0:
                con_m.append("g")
            elif self.condensations[i][2] > 0:
                con_m.append("b")
            elif self.condensations[i][2] < 0:
                con_m.append("r")
        ax.scatter(con_x, con_y, color=con_m)
        ax.set_title('Gas density at plane of z = ' + str(z))
        ax.set_xlabel('x (pc)')
        ax.set_ylabel('y (pc)')
        cbar.set_label('number density ($cm^{-3})$', rotation=270, labelpad=16)
        if save_as is not None:
            plt.savefig(save_as)
        else:
            plt.show()



#mc = Molecular_cloud()
#mc.set_molecular_cloud_parameters(200, 500000, 2400)
#mc.get_crossing_time(3)
#mc.set_r_0(0, override_radius=3)
#print(mc.lookup_number_density(-1.2323, 0.0001, 0.002, 0, convert_units=False))
#print(mc.get_number_density(-1.2323, 0.0001, 0.002, 0, convert_units=False))
#mc.add_condensation(-2, 2, 2, 342, 2)
#mc.add_condensation(-0, 0, 0, 342, 2)
#mc.add_condensation(2, -2, -2, 342, 2)
#print(mc.condensations)
#mc.set_molecular_cloud_parameters(1e5, 1e7, 840)
#print(mc.find_radius_of_cloud(0.05, show_plot=True))
#print(mc.get_crossing_time(mc.find_radius_of_cloud(0.05)))
#print(mc.galactic_tide(10))
#print(mc.get_mean_molecular_mass())
#mc.plot_number_density(z=0, scale=10)
#print(mc.get_molecular_cloud_mass_box(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
#print(mc.get_molecular_cloud_mass(0, 0, 0, 0.5))
#print(mc.get_molecular_cloud_mass(-6.5, 6.5, -6.5, 6.5, -6.5, 6.5))
#print(mc.plummer_sphere_g(20, 0, 0, 0))
#mc.plot_path(int(2*855780), z=0, scale=10, save_as="/tmp/test.png")