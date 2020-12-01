# coding: utf8

import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import csv
from matplotlib.collections import LineCollection
import imageio
import os
import shutil
import uuid
from scipy import optimize
import bisect
import time
from mpl_toolkits.axisartist.axislines import Subplot

# I've made these fairly large so that they are readable on a printed page with 6 subfigures
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


unpack_header = None
unpacked_data = None
unpack_data_lines_range = None
time_since_unpack_used = time.time()
start_index = 0

class CustomException(Exception):
    """Allows for a custom exception to be raised."""
    pass


class Plotter:
    """Class for various plotting tools.

    Can use either raw data from a simulation, or saved data from a csv.

    Typical usage:

        foo = integrate_simulation() p = data.Plotter(foo) p.plot_aei(show=True,
        save_as="Figure1.png", alim=(0,80))

        foo = ["particle_1_uuid", "particle_2_uuid"] p = data.Plotter(foo)
        p.plot_aei()
    """

    def __init__(self, data, location=""):
        """Constructor for Plotter object.

        Args:
            data: list of Particle objects.
            location: directory that the particles data is saved in. Leave blank
                to use the temporary directory.
        """
        self.particles = None
        self.a = []
        self.e = []
        self.inc = []
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.orbit_plot_index = 0

        if type(data) is str:
            self.particles = [Particle(label=data, location=location)]
        elif type(data) is list:
            if type(data[0]) is str:
                self.particles = []
                for lbl in data:
                    self.particles.append(Particle(label=lbl, location=location))
            elif type(data[0]) is Particle:
                self.particles = data
            else:
                raise CustomException("Plotter data must be str, list of str, or list of Particle. Found list of " +
                                      str(type(data[0])))
        else:
            raise CustomException(str(type(data)) + " is not a valid data type for Plotter.")

        self._meets_delta_a_threshold_max_change = None
        self._meets_delta_a_threshold_min_change = None


    def read_data_aei(self):
        global unpack_header, unpacked_data
        """Reads in aei and t data from each particle"""
        print("This is an old method of accessing particle data and will be removed. Please access the particle object directly with self.particles[index].get_<x>()")
        self.a = np.zeros([len(self.particles)]).tolist()
        self.e = np.zeros([len(self.particles)]).tolist()
        self.inc = np.zeros([len(self.particles)]).tolist()
        for i in range(len(self.particles)):
            if i % 100 == 0:
                print("reading particle", str(i)+"/"+str(len(self.particles)))
            self.a[i] = self.particles[i].get_a()
            self.e[i] = self.particles[i].get_e()
            self.inc[i] = self.particles[i].get_inc()
            self.t = self.particles[i].get_t()
        unpacked_data = None  # Clear memory
        unpack_header = None

    def read_data_xyz(self):
        """Reads in xyz and t data from each particle"""
        self.x = np.zeros([len(self.particles)]).tolist()
        self.y = np.zeros([len(self.particles)]).tolist()
        self.z = np.zeros([len(self.particles)]).tolist()
        for i in range(len(self.particles)):
            self.x[i] = self.particles[i].get_x()
            self.y[i] = self.particles[i].get_y()
            self.z[i] = self.particles[i].get_z()
            self.t = self.particles[i].get_t()

    def plot_size_distribution(self, k0, k1, ymax=200, save_as="", ejected_ylim=None, remaining_ylim=None, show_count=True, ejected_max_bin=None):
        ''' your_bins=20
        data=[]
        arr=plt.hist(data,bins=your_bins)
        for i in range(your_bins):
            plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
        '''
        e_threshold = 1
        ejected_radii = []
        fig, axs = plt.subplots(2, 1)
        axs[0].grid(color='lightgrey', alpha=0.5)
        axs[1].grid(color='lightgrey', alpha=0.5)
        fig.set_size_inches(12.8, 9.2)
        n = 0
        radii0 = []
        radii1 = []
        pre_ejec = []
        for i, particle in enumerate(self.particles):
            if i % 500 == 0:
                print(i/len(self.particles))
            if particle.get_e()[k0] < e_threshold:
                radii0.append(float(particle.get_radius()))
                n += 1
            else:
                pre_ejec.append(particle)
            if particle.get_e()[k1] < e_threshold:
                radii1.append(float(particle.get_radius()))
            elif particle.get_e()[k1] >= e_threshold and particle not in pre_ejec:
                ejected_radii.append(float(particle.get_radius()))
            else:
                pass
                #print("Warning: particle ejected at first index.")
            particle.clear()
        bins = np.linspace(0.1, np.ceil(float(max(radii0))), 20)
        if ejected_max_bin is None:
            bins_ejected = np.linspace(0.1, np.ceil(float(max(ejected_radii))), 20)
        else:
            bins_ejected = np.linspace(0.1, ejected_max_bin, 20)
        axs[0].hist(radii0, bins=bins, color='deepskyblue', label="Initial", rwidth=1, align="mid", edgecolor="k")
        axs[0].hist(radii1, bins=bins, color='#80dfff', rwidth=1, label="Final", align='mid', edgecolor='k')
        ejec = axs[1].hist(ejected_radii, bins=bins_ejected, color='#AC58FA', label="Ejected Particles", rwidth=1, align="mid", edgecolor="k")
        if show_count:
            for i in range(len(bins_ejected)-1):
                axs[1].text(ejec[1][i], ejec[0][i], str(ejec[0][i]))
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        if remaining_ylim is not None:
            axs[0].set_ylim(0.1, remaining_ylim)
        if ejected_ylim is not None:
            axs[1].set_ylim(0.1, ejected_ylim)
        #plt.ylim(0.1, ymax)
        #plt.title("Histogram of sizes at t = "+str(t0)+" and "+str(t1))
        axs[0].set_xlabel("R of remaining [km]")
        axs[0].set_ylabel("N Remaining")
        axs[0].legend()
        axs[1].set_xlabel("R of ejected [km]")
        axs[1].set_ylabel("N Ejected")
        lines = []
        for i in range(1, len(bins_ejected)-1):
            lines.append("Bin " + str(i) + " ranges ["+str(bins_ejected[i-1])+", " + str(bins_ejected[i])+")\n")
            ejec_count = 0
            total_count = 0
            for j in range(len(radii0)):
                if bins_ejected[i-1] <= radii0[j] < bins_ejected[i]:
                    total_count +=1
                if j < len(ejected_radii):
                    if bins_ejected[i-1] <= ejected_radii[j] < bins_ejected[i]:
                        ejec_count += 1
            if total_count > 0:
                lines.append(str(ejec_count/total_count * 100)+"% ejected from bin" + str(i)+"\n")
            lines.append("This bin contains: " + str(total_count) + " items\n\n")
        lines.append("total particles at start index: "+str(len(radii0))+"\n")
        lines.append("total ejected is "+str(len(ejected_radii))+"\n")
        plt.tight_layout()
        if len(radii0) > 0:
            lines.append(str(len(ejected_radii)/len(radii0)*100) + "% ejected")
        if save_as == "":
            plt.show()
        else:
            print("Saving")
            f = open(save_as+".txt", "w")
            f.writelines(lines)
            f.close()
            plt.savefig(save_as, dpi=600)
        plt.close()

    def fit_size_dist(self, index, save_as=None):
        radii = []
        for i, particle in enumerate(self.particles):
            if particle.get_e()[index] < 1:
                radii.append(particle.get_radius())
        sd = Size_Distribution()
        sd.fit_size_distribution(radii)
        sd.plot_size_distribution(save_as=save_as)


    def save_ejected_particle_list(self, index, file):
        """ This saves a text file with a list of all the particles that are ejected at the
        specified index.

        :param index: index to test for ejection at
        :param file: file path to save to
        :return: nothing :(
        """
        ejected_particles = []
        try:
            for particle in self.particles:
                if particle.get_e()[index] >= 1:
                    ejected_particles.append(str(particle.get_label())+"\n")
                particle.clear()
        except StopIteration:
            pass
        f = open(file, "w")
        f.writelines(ejected_particles)
        f.close()

    def plot_ejections_by_a(self, start_index=0, save_as=None, max_a=500000):
        a = []
        if start_index != 0:
            start_index -= 1
        for particle in self.particles:
            if particle.get_e()[len(particle.get_e())-1] >= 1 and particle.get_e()[start_index] < 1:
                a.append(particle.get_a()[start_index])
            particle.clear()
        print(a)
        plt.hist(a, bins=20)
        plt.xlabel("Initial a [au]")
        plt.ylabel("Number ejected from a bin")
        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as, dpi=600)

    def plot_ejection_times(self, save_as=None, start_index=0):
        """ Plots a histogram of the time range in which a particle is ejected.

        :param save_as: path to save file to.
        """
        ejected_times = []
        if start_index != 0:
            start_index -= 1
        for particle in self.particles:
            for j in range(start_index, len(particle.get_t())):
                if particle.get_e()[j] >= 1:
                    if j == start_index:
                        break
                    ejected_times.append(particle.get_t()[j])
                    break  # breaks the inner loop so we go to the next particle, not next time step of same one.
            particle.clear()

        #print(ejected_times)
        fig = plt.figure()
        ax = Subplot(fig, 111)
        fig.add_subplot(ax)
        fig.set_size_inches(12.8*0.8, 9.2*0.8)
        ax.hist(ejected_times, bins=100)
        # This was just to draw a line a index 118 for the report
        # x = np.ones(100)*particle.get_t()[118]
        x2 = np.ones(100) * 1266240.9270667
        ylim = ax.get_ylim()
        y = np.linspace(0, ylim[1], 100)
        ax.set_ylim(ylim)
        #ax.plot(x, y, color='black', linestyle="-")
        ax.plot(x2, y, color='black', linestyle="--")
        ax.grid()
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.linspace(start, end, 5))
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel("Number of particles ejected")
        if save_as is not None:
            plt.savefig(save_as, dpi=600)
        else:
            plt.show()
        plt.close()



    def eccentricity_stats(self, t1, t2):
        e_diff = []
        r_particle = []
        count_increase = 0
        for particle in self.particles:
            delta = particle.get_e()[t2] - particle.get_e()[t1]
            e_diff.append(delta)
            r_particle.append(np.round(float(particle.get_radius()), decimals=3))
            if delta > 0:
                count_increase += 1
        y_centerline = np.zeros(10)
        x_centerline = np.linspace(0, np.amax(r_particle), 10)
        print("The mean change in e is", np.mean(e_diff))
        print(count_increase, "particles had an overall increase in eccentricity")
        print(str(len(e_diff) - count_increase), "particles had an overall decrease or no change")
        plt.scatter(r_particle, e_diff)
        plt.plot(x_centerline, y_centerline, ls='--', color='0.5')
        plt.xlabel("particle radius [km]")
        plt.ylabel("change in e")
        plt.show()

    def e_a_changes_hist(self, t1, t2):
        delta_e = []
        delta_a = []
        for particle in self.particles:
            delta_e.append(particle.get_e()[t2] - particle.get_e()[t1])
            delta_a.append(particle.get_a()[t2] - particle.get_a()[t1])
        fig, axs = plt.subplots(2, 1)
        axs[0].hist(delta_e)
        axs[0].set_xlabel("$\Delta e$")
        axs[0].set_ylabel("Count")
        axs[1].hist(delta_a)
        axs[1].set_xlabel("$\Delta a$")
        axs[1].set_ylabel("Count")
        plt.show()


    def plot_aei_dots(self, index, save_as=None, alim=None):
        self.read_data_aei()
        i = []
        a = []
        e = []
        for k in range(len(self.a)):
            if self.e[k][index] < 1.0:
                i.append(self.inc[k][index] * (180 / np.pi))  # convert to degrees
                a.append(self.a[k][index])
                e.append(self.e[k][index])
        print("number of bound particles at index", str(index), "is", str(len(i)))
        fig, axs = plt.subplots(2, 1)
        axs[0].grid(color='lightgrey', alpha=0.5)
        axs[1].grid(color='lightgrey', alpha=0.5)
        axs[0].scatter(a, i, marker='.', color='black', s=1)
        axs[1].scatter(a, e, marker='.', color='black', s=1)
        axs[0].set_ylabel("Inclination, i")
        axs[0].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.01f}°"))
        axs[1].set_ylabel("Eccentricity, e", labelpad=20)
        axs[1].set_xlabel("Semi-major axis, a [au]")
        if alim is not None:
            axs[0].set_xlim((1500, alim))
            axs[1].set_xlim((1500, alim))
        if save_as is not None:
            plt.savefig(save_as, dpi=600)
        else:
            plt.show()
        plt.close()

    def meets_delta_a_threshold(self, comet_index, threshold):
        if self._meets_delta_a_threshold_max_change is None:
            delta = []
            for particle in self.particles:
                delta.append(abs(particle.get_a()[len(particle.get_a())-1] - particle.get_a()[0]))
            self._meets_delta_a_threshold_max_change = np.mean(delta)
            print(self._meets_delta_a_threshold_max_change)
        if (self.particles[comet_index].get_a()[len(self.a[comet_index])-1] - self.particles[comet_index].get_a()[0]) >= threshold*self._meets_delta_a_threshold_max_change:
            return True
        else:
            return False



    def plot_aei(self, save_as=None, show=True, gradient=True, colourmap="viridis", alim=None, add_particle_labels=True, delta_a_threshold=0, subsample=1, log_scale=False, start_index=0):
        global unpack_header, unpacked_data
        """Plots an aei chart of the objects data fields.

        Args:
            save_as: file path to save the plot.
            show: boolean, show a GUI plot when executing?
            gradient: Represent time with a gradient of colour.
            colourmap: The colourmap to use for the gradient (Ensure it is
                Perceptually uniform).
            subsample: how often do we sample the data? 1 = sample every point. 4 = sample every 4th point, etc.
                This should generally be used for large simulations to reduce the number of dots drawn.
            delta_a_threshold: show particles with changes in a > delta_a_threshold*mean_change, i.e. a value
                of 1 shows only particles with changes greater than the mean.
            alim: The graphing limits on the semi-major axis, should be tuple
                like (min, max).
        """
        t1 = time.time()
        #if len(self.a) == 0:
        #    print("reading data")
        #    self.read_data_aei()
        #    print(".. Done")
        if time_since_unpack_used > time.time()+1:
            unpack_header = None
            unpacked_data = None
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(12.8, 9.2)
        i_lims = [0, 1]
        a_lims = [0, 1]
        e_lims = [0, 1]
        set_lims = False
        added_colour_bar = False
        #plt.rcParams.update({'font.size': 32})
        for i, particle in enumerate(self.particles):
            if i % 100 == 0:
                print("processing particle "+str(i)+" of "+str(len(self.particles)))
                if time_since_unpack_used > time.time() + 1:
                    unpack_header = None  # This is to clear memory.
                    unpacked_data = None
            if gradient and (delta_a_threshold == 0 or self.meets_delta_a_threshold(i, delta_a_threshold)):
                #Gradient lines: https://matplotlib.org/examples/pylab_examples/multicolored_line.html
                # Note, the subsample variable sets the increment, so subsample = 1 means don't skip anything,
                # subsample = 4 means only sample every 4th datapoint
                ejection_point = []
                last_bound = len(particle.get_t())-1
                for j in range(start_index, len(particle.get_t())):
                    if particle.get_e()[j] < 1:
                        pass
                    else:
                        ejection_point = [particle.get_a()[j], particle.get_e()[j], particle.get_inc()[j] * (180/np.pi)]
                        last_bound = j
                        break
                if last_bound > 0:
                    #print("diff: "+str((particle.get_a()[0] - particle.get_a()[last_bound])))
                    if np.abs(particle.get_a()[0] - particle.get_a()[last_bound]) < 100:
                        axs[0].scatter(particle.get_a()[0], particle.get_inc()[0] * (180/np.pi), marker='.', s=1,
                                       color='k')
                        axs[1].scatter(particle.get_a()[0], particle.get_e()[0], marker='.', s=1, color='k')
                    else:
                        times = np.array(particle.get_t()[start_index:last_bound:subsample])
                        inc = np.array(particle.get_inc()[start_index:last_bound:subsample]) * (180/np.pi)
                        a = np.array(particle.get_a()[start_index:last_bound:subsample])
                        e = np.array(particle.get_e()[start_index:last_bound:subsample])

                        if len(times) > 0:

                            points_ai = np.array([a, inc]).T.reshape(-1, 1, 2)
                            segments_ai = np.concatenate([points_ai[:-1], points_ai[1:]], axis=1)

                            # Colour maps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
                            # Check for "Perceptually uniform"
                            lc_ai = LineCollection(segments_ai, cmap=plt.get_cmap(colourmap),
                                                norm=plt.Normalize(0, times[len(times) - 1]))

                            lc_ai.set_array(times)
                            lc_ai.set_linewidth(2)

                            points_ae = np.array([a, e]).T.reshape(-1, 1, 2)
                            segments_ae = np.concatenate([points_ae[:-1], points_ae[1:]], axis=1)

                            lc_ae = LineCollection(segments_ae, cmap=plt.get_cmap(colourmap),
                                                norm=plt.Normalize(0, times[len(times) - 1]))

                            lc_ae.set_array(times)
                            lc_ae.set_linewidth(2)

                            axs[0].add_collection(lc_ai)
                            axs[1].add_collection(lc_ae)
                            #print("added collection")

                # Add 'X' for the ejected particles at the ejection point (because they can go very far out and distort
                # the plots scale if we draw their path after ejection)
                #if len(ejection_point) > 0:
                #    axs[0].scatter(ejection_point[0], ejection_point[2], marker='x', color='k')
                #    axs[1].scatter(ejection_point[0], ejection_point[1], marker='x', color='k')

                # Using this method to draw to a plot removes the ability to auto-set the axis limits
                # and it default to 1,1 - which is generally outside of the data's range. We correct here:
                if np.abs(particle.get_a()[0] - particle.get_a()[last_bound]) >= 100:
                    if len(inc) > 0:
                        if (not set_lims) or (np.amin(inc) < i_lims[0]):
                            i_lims[0] = np.amin(inc)
                        if (not set_lims) or (np.amax(inc) > i_lims[1]):
                            i_lims[1] = np.amax(inc)
                        if (not set_lims) or (np.amin(a) < a_lims[0]):
                            a_lims[0] = np.amin(a)
                        if (not set_lims) or (np.amax(a) > a_lims[1]):
                            a_lims[1] = np.amax(a)
                        if (not set_lims) or (np.amin(e) < e_lims[0]):
                            e_lims[0] = np.amin(e)
                        if (not set_lims) or (np.amax(e) > e_lims[1]):
                            e_lims[1] = np.amax(e)
                            set_lims = True

                if np.abs(particle.get_a()[0] - particle.get_a()[last_bound]) >= 100:
                    if not added_colour_bar and len(times) > 0: # The colourbar is the same for all particles, so this makes sure we don't duplicate it.
                        fig.subplots_adjust(right=0.8)
                        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                        cbar1 = fig.colorbar(lc_ai, cax=cbar_ax)
                        cbar1.set_label('Time [years]', rotation=270, labelpad=24)
                        added_colour_bar = True
                    #cbar2 = fig.colorbar(lc_ae, ax=axs[1])
                    #cbar2.set_label('Time [years]', rotation=270, labelpad=16)
            elif (delta_a_threshold == 0 or self.meets_delta_a_threshold(i, delta_a_threshold)):
                axs[0].plot(particle.get_a()[i], particle.get_inc()[i])
                axs[1].plot(particle.get_a()[i], particle.get_e()[i])
            if gradient:
                axs[0].set_ylim(i_lims[0], i_lims[1])
                axs[0].set_xlim(a_lims[0], a_lims[1])
                axs[1].set_ylim(e_lims[0], e_lims[1])
                axs[1].set_xlim(a_lims[0], a_lims[1])
        particle.clear() # discards particle data, this means it needs to be re-read from file but stops excesive
        # memory usages for large data sets.

        axs[0].set_ylabel("Inclination, i")
        axs[0].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.01f}°"))
        axs[0].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.01f}°"))

        if alim is not None:
            if type(alim) is int:
                axs[0].set_xlim((1500, alim))
                axs[1].set_xlim((1500, alim))
            else:
                axs[0].set_xlim(alim)
                axs[1].set_xlim(alim)

        # label data
        if add_particle_labels:
            for i in range(len(self.particles)):
                axs[0].annotate(str(particle.get_label())[:5], xy=(particle.get_a()[i][0], particle.get_inc()[0]))
                axs[1].annotate(str(particle.get_label())[:5], xy=(particle.get_a()[0], particle.get_e()[0]))

        axs[1].set_ylabel("Eccentricity, e", labelpad=20)
        if log_scale:
            axs[1].set_xscale('log')
            axs[0].set_xscale('log')
        axs[1].set_xlabel("Semi-major axis, a [au]")
        axs[0].grid(color='lightgrey', alpha=0.5)
        axs[1].grid(color='lightgrey', alpha=0.5)
        print("Saving/showing")
        if save_as is not None:
            plt.savefig(save_as, dpi=600, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        t2 = time.time()
        print("Finished in", str(t2 - t1), "s")

    def plot_orbit(self, index, scale, plot_backwards=1000000, show=False):
        """Plots particle orbits based on saved x,y,z positions. Uses color
        gradient to represent z.

        Note: The current position is always coloured black, the trails are
        colour coded by z.

        Returns: Path to saved plot.

        Args:
            index: Index of data to plot.
            scale: Distance, in AU, to plot out from the centre.
            plot_backwards: numbers of time steps back from the present index to
                plot as a trailing line.
            show: Display plot to screen, default=False.
        """
        self.read_data_xyz()
        x = []
        y = []
        z = []
        fig = plt.subplot()
        for i in range(len(self.particles)):
            x.append(self.x[i][index])
            y.append(self.y[i][index])
            z.append(self.z[i][index])
            fig.annotate(str(self.particles[i].get_label())[:6], xy=(x[i], y[i]))
            xt = []
            yt = []
            zt = []
            for j in range(np.minimum(plot_backwards, index)):
                if (j % np.floor(np.minimum(plot_backwards, index)/10)) == 0:  # subsample
                    xt.append(self.x[i][index - j])
                    yt.append(self.y[i][index - j])
                    zt.append(self.z[i][index - j])
            # display z data as color code
            ztnp = np.array(zt)
            points = np.array([xt, yt]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap("viridis"), norm=plt.Normalize(-scale, scale), alpha=0.6)
            lc.set_array(ztnp)
            lc.set_linewidth(2)
            fig.add_collection(lc)
            if i == 0:
                cbar = plt.colorbar(lc)
                cbar.set_label('z [A.U.]', rotation=270)
            #fig.plot(xt, yt, color='black', alpha=0.2)

        fig.scatter(x, y, marker='.', color='black')
        fig.scatter(0, 0, marker='*', color='orange')  # plot the sun
        plt.xlim(-scale, scale)
        plt.ylim(-scale, scale)
        plt.xlabel('x [A.U.]')
        plt.ylabel('y [A.U.]')
        if show:
            plt.show()
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")
        save_path = os.path.join(".tmp", "orbit_plot_for_index_"+str(index)+".png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    @staticmethod
    def save_as_gif(images, save_as, move_raw=False, make_gif=True):
        """Combines an array of image files into a gif animation. Designed for
        saving orbit plots as time animation, but could be used for any gif
        creation.

        Args:
            images: list of images to use as the individual frames.
            save_as: Path to save gif animation to. When saving raw images a
                directory of this name is used with an additional _raw added,
                images are saved in this directory with their index number as
                their filename.
            move_raw: Boolean, moves all raw images into new directory, e.g.
                move orbit plots from the .tmp directory.
            make_gif: Create a gif or not. This is an option so that I can use the
                same function to move image files around in bulk without actually
                creating the gif.
        """
        # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        if make_gif:
            with imageio.get_writer(save_as+".gif", mode='I') as writer:
                for img in images:
                    image = imageio.imread(img)
                    writer.append_data(image)
        if move_raw:
            if not os.path.exists(save_as+"_raw"):
                os.mkdir(save_as+"_raw")
            for i in range(len(images)):
                shutil.copyfile(img, os.path.join(save_as+"_raw", str(i)+".png"))
        # Clean up - delete the tmp files
        shutil.rmtree(".tmp")


pack_threshold = 1000000
pack_data = None
pack_count = 0
pack_tmp_file = ".tmp/particles_"+str(time.time())+".dat"


def pack_into_file(id, data, eof=False):
    """This combines multiple csv files into 1.

    Params:
        id: The file name it would have if stored as a single file
        data: One line of csv data
    """
    global pack_data, pack_count, pack_tmp_file
    if pack_data is None:
        pack_data = np.zeros(pack_threshold).tolist()
    pack_data[pack_count] = str(id)+":{"+data+"}\n"
    pack_count += 1
    if pack_count >= pack_threshold:
        m = "w"
        if os.path.exists(pack_tmp_file): m = "a"
        f = open(pack_tmp_file, m)
        f.writelines(pack_data)
        f.close()
        pack_count = 0
        print("Wrote out packing file")
    elif eof:
        m = "w"
        if os.path.exists(pack_tmp_file): m = "a"
        f = open(pack_tmp_file, m)
        for i in range(0, pack_count):
            f.write(pack_data[i])
        f.close()





def unpack_with_header(id):
    global unpack_header, unpacked_data, time_since_unpack_used, unpack_data_lines_range, start_index
    time_since_unpack_used = time.time()
    if unpack_header is None:
        f = open(pack_tmp_file+".header", "r")
        unpack_header = f.read().split("\n")
        f.close()
        print("read header into memory")
    if unpacked_data is None:
        # This starts it off with ~10 particles in memory,
        # the idea is it will dynamically read more as needed.
        unpack_data_lines_range = range(0, 5020)
        start_index = 0
        unpacked_data = []
        with open(pack_tmp_file) as fp:
            for i, line in enumerate(fp):
                if i in unpack_data_lines_range:
                    unpacked_data.append(line)
                elif i > 5020:
                    break
        print("read data up to line 5020")
    data = []
    for i in range(len(unpack_header) - 1):
        line = unpack_header[i].split()
        if line[0] == id:
            index1, index2 = int(line[1]), int(line[2])
            # first check we have these file lines in memory
            if index1 in unpack_data_lines_range and index2 in unpack_data_lines_range:
                # Package the data
                # print("found")
                data = unpacked_data[index1-start_index:index2 - start_index + 1]
            else:
                print("reading in more of the file")
                # Get the relevant file lines
                unpack_data_lines_range = range(index1, index1 + 5000*(index2-index1))
                start_index = index1
                unpacked_data = []
                with open(pack_tmp_file) as fp:
                    for j, line in enumerate(fp):
                        if j in unpack_data_lines_range:
                            unpacked_data.append(line)
                        elif j > index1 + 5000*(index2-index1):
                            break
                data = unpacked_data[index1 - start_index:index2 - start_index + 1]
    '''
    if unpacked_data is None:
        f = open(pack_tmp_file, "r")
        unpacked_data = f.read().split("\n")
        f.close()
        print("Read", str(len(unpacked_data)), "lines into memory")
    '''
    if len(data) == 0:
        print("Error, cannot find particle "+id)
    return data



def unpack_from_file(id):
    """ Returns a full csv string from a multi-csv pack file"""
    global unpacked_data
    data = []
    found_index = []
    skip = 1
    if os.path.exists(pack_tmp_file+".header"):
        #print("using repacked with header format")
        return unpack_with_header(id)
    if unpacked_data is None:
        f = open(pack_tmp_file, "r")
        unpacked_data = f.read().split("\n")
        f.close()
        print("Read", str(len(unpacked_data)), "lines into memory")
    for i in range(len(unpacked_data)):
        if unpacked_data[i].split(":{")[0] == id:
            data.append(unpacked_data[i].split(":{")[1].strip("}"))
            """ This is to look for patterns in the data that make it faster to read, specifically, each particle is 
            most likely to be on line N*n+c, where N is total number of particles, n is particle number and c is const.
            This looks for that pattern by finding a 'skip' value, once confident, it breaks out of this loop and
            resumes searching of the particle id, but skipping the 'skip' value number of lines between searches.
            """
            found_index.append(i)
            #print(found_index)
            n = len(found_index)
            if n > 2:
                if found_index[n-1] - found_index[n-2] == found_index[n-2] - found_index[n-3] != 1:
                    skip = found_index[n-1] - found_index[n-2]
                    #print("found skip of "+str(skip))
                    break
    if skip != 1:
        for i in range(found_index[len(found_index)-1], len(unpacked_data), skip):
            if unpacked_data[i].split(":{")[0] == id:
                data.append(unpacked_data[i].split(":{")[1].strip("}"))
    #print("data length is "+str(len(data))+" lines")
    return data

class Particle:
    """Stores data for a particle.

    Data is streamed to a file, to allow for long simulation and reduce
    memory usage. There is no need to give a file location as a temporary file
    will be created, save_csv will later move the file to a desired permanent
    location.

    typical usage:

        p = data.Particle()

        # Using a particle to save data from a simulation
        p.open_stream('w')
        while simulation is running generate new a, e, i, t, x, y, z data for particle:
            p.stream_in(a, e, i, t, x, y, z)
        p.close_stream()
        p.save_csv('permanent/location')

        # Reading from a particle file.
        p = data.Particle(location='permanent/location', label='Particle_uuid')
        p.open_stream('r')
        plt.plot(p.get_a(), p.get_e())
        p.close_steam()
    """
    def __init__(self, label=None, location=".tmp", pack_files=True):
        """

        Args:
            label: An identifier for the particle. Leave empty for a new particle and a UUID will be given.
            location: Location particle data is stored. Of loading an existing particle, this is the directory to load
                    it from. If making a new particle leave this empty.
        """
        # Meta data data in black by default and must be set by set_metadata(type, r, density, m)
        self.type = "n/a"
        self.radius = 0
        self.density = 0
        self.mass = 0
        # Standard stuff
        self.a = []
        self.inc = []
        self.e = []
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.mode = None
        self.location = location
        if location != ".tmp" and not os.path.exists(os.path.join(location, "particles.dat")):
            self.use_packing_file = False
        else:
            self.use_packing_file = pack_files
        if label is None:
            # create a unique id if label is not set
            self.label = str(uuid.uuid1())
        else:
            self.label = label
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

    def set_metadata(self, type_, radius, density, mass, label=None):
        """Sets metadata about the particle.

        Args:
            type_: String - "comet" or "asteroid"
            radius: radius in km of the object
            density: density in km/m^3 of the object
            mass: mass of the object in kg
        """
        if label is not None:
            assert label == self.label
        self.type = type_
        self.radius = float(radius)
        self.density = float(density)
        self.mass = float(mass)

    def get_radius(self):
        return self.radius

    def get_type(self):
        return self.type

    def get_density(self):
        return self.density

    def get_mass(self):
        return self.mass

    def get_label(self):
        """Returns a unique identifier for the particle."""
        return self.label

    def open_stream(self, mode, label=None):
        """
        Args:
            mode:
        """
        global pack_tmp_file
        if label is not None and label != self.label:
            pass
            # print("Labels do not match, saving on behalf another particle")
        else:
            label = self.label
        if mode == 'w':
            if self.use_packing_file:
                pack_into_file(label, "type radius density mass")
                pack_into_file(label, self.type+" "+str(self.radius)+" "+str(self.density)+" "+str(self.mass))
                pack_into_file(label, 'a e i t x y z')
            else:
                self.csvfile = open(os.path.join(self.location, label + ".csv"), mode, newline='\n')
                self.writer = csv.writer(self.csvfile, delimiter=' ')
                self.writer.writerow(['type', 'radius', 'density', 'mass'])
                self.writer.writerow([self.type, self.radius, self.density, self.mass])
                self.writer.writerow(['a', 'e', 'i', 't', 'x', 'y', 'z'])
        elif mode == 'a':
            if self.use_packing_file:
                pass
            else:
                self.csvfile = open(os.path.join(self.location, str(label) + ".csv"), mode, newline='\n')
                self.writer = csv.writer(self.csvfile, delimiter=' ')
        elif mode == 'r':
            if self.use_packing_file:
                if self.location != ".tmp":
                    pack_tmp_file = os.path.join(self.location, "particles.dat")
                else:
                    # If there is anything on the buffer, we need to write out the buffer before reading.
                    if pack_count > 0:
                        pack_into_file("eof", "goodbye", eof=True)
                self.csvfile = unpack_from_file(label)
            else:
                self.csvfile = open(os.path.join(self.location, str(label) + ".csv"), mode, newline='\n')
            self.reader = csv.reader(self.csvfile, delimiter=' ')
            # If the file contains metadata, the metadata header contains the word "type", so the metadata
            # reader will trigger. If it's an old file, it will just skip the header line and get to data.
            try:
                if "type" in next(self.reader):
                    md = next(self.reader)
                    self.set_metadata(md[0], md[1], md[2], md[3])
                    #next(self.reader)  # skip the data's header
            except StopIteration:
                print("StopIteration")
                print(self.get_label())
        else:
            raise NotImplementedError
        self.mode = mode
        # Note I am purposely not closing the file so that we can continue to write to it.

    @staticmethod
    def check_var(value, t):
        """Checks the variable is within a valid range.

        Args:
            value: Value of variable.
            t: What type of variable is it, determined the expected range.
        """
        assert type(value) is float
        assert type(t) is str
        if t == 'a':
            assert value >= 0
        elif t == 'e':
            assert value >= 0
            assert value <= 1.5  # This may need to be adjusted
        elif t == 'i':
            assert value >= 0
            assert value <= 180
        elif t == 't':
            assert value >= 0
        else:
            print("Warning, no tests are in place for type", t)
        # todo: add limits for x,y,z

    def save_next(self, next_a, next_e, next_i, next_t, next_x, next_y, next_z, label=None):
        """Saves the parameters provided to the open file.

        Args:
            next_a: semi-major axis
            next_e: eccentricity
            next_i: inclination
            next_t: time
            next_x: x position
            next_y: y position
            next_z: z position
        """
        #self.check_var(next_a, "a")
        if label is not None and label != self.label:
            pass
            # print("Labels do not match, saving on behalf of another particle")
        else:
            label = self.label
        if self.mode == 'w' or self.mode == 'a':
            if self.use_packing_file:
                pack_into_file(label, str(next_a)+" "+str(next_e)+" "+str(next_i)+" "+str(next_t)+" "+str(next_x)+" "+str(next_y)+" "+str(next_z))
            else:
                self.writer.writerow([next_a, next_e, next_i, next_t, next_x, next_y, next_z])
        else:
            raise CustomException("Particle cannot write data as it is in the incorrect mode")

    def close_stream(self):
        """Closes the file"""
        if not self.use_packing_file:
            self.csvfile.close()
        self.mode = None

    def read_in(self):
        """Read in the values of a,e,i into memory."""
        self.clear()
        self.open_stream('r')
        for row in self.reader:
            if row[0] != "a":
                self.a.append(float(row[0]))
                self.e.append(float(row[1]))
                self.inc.append(float(row[2]))
                self.t.append(float(row[3]))
        self.close_stream()

    def clear(self):
        """Clears data from memory, note, data is saved to files so this doesn't
        delete anything
        """
        self.a = []
        self.inc = []
        self.e = []
        self.t = []
        self.x = []
        self.y = []
        self.z = []

    def read_in_xyz(self):
        """Read in values x,y,z into memory. The idea of having 3 difference
        read_in methods is to limit memory usage.
        """
        self.clear()
        self.open_stream('r')
        for row in self.reader:
            if row[0] != "a":
                try:
                    self.x.append(float(row[4]))
                    self.y.append(float(row[5]))
                    self.z.append(float(row[6]))
                except IndexError as e:
                    print("Particle", self.label, "does not contain the requested data. Is it loaded from an old file?")
                    print(e)
        self.close_stream()

    def get_a(self):
        """Returns the semi-major axis.

        Raises: CustomException, if the data field is empty.

        Returns:
            An array of semi-major axis values.
        """
        if len(self.a) == 0:
            self.read_in()
        if self.a is not None:
            return self.a
        else:
            CustomException("Particle asked for result that has not been set.")

    def get_e(self):
        """Returns the eccentricity.

        Raises: CustomException, if the data field is empty.

        Returns:
            An array of eccentricity values.
        """
        if len(self.e) == 0:
            self.read_in()
        if self.e is not None:
            return self.e
        else:
            CustomException("Particle asked for result that has not been set.")

    def get_ejected(self, index):
        """ Returns True if eccentricity >= 1 at the specified index.
        Use index=-1 to access the last index.
        """
        if index == -1:
            index = len(self.get_e())-1
        try:
            return self.get_e()[index] >= 1
        except IndexError:
            print("Warning, couldn't find particle data.")
            return True

    def get_inc(self):
        """Returns the inclination.
            self.read_in()

        Raises: CustomException, if the data field is empty.

        Returns:
            An array of inclination values.
        """
        if len(self.inc) == 0:
            self.read_in()
        if self.inc is not None:
            return self.inc
        else:
            CustomException("Particle asked for result that has not been set.")

    def get_t(self):
        """Returns the times.

        Raises: CustomException, if the data field is empty.

        Returns:
            An array of time values.
        """
        if len(self.t) == 0:
            self.read_in()
        if self.t is not None:
            return self.t
        else:
            CustomException("Particle asked for result that has not been set.")

    def get_x(self):
        """accessor for the particles x position."""
        if len(self.x) == 0:
            self.read_in_xyz()
        return self.x

    def get_y(self):
        """Accessor for the particle's y position."""
        if len(self.y) == 0:
            self.read_in_xyz()
        return self.y

    def get_z(self):
        """Accessor for the particle's z position"""
        if len(self.z) == 0:
            self.read_in_xyz()
        return self.z

    def save_csv(self, save_as):
        """Saves a csv of the data stored in the object it is called on.

        Args:
            save_as: The file path to save to, note the '.csv' extension is
                automatically added.
        """
        global pack_tmp_file
        # all we need to do here is move the existing csv from .tmp to the desired location
        if self.use_packing_file:
            if pack_tmp_file != os.path.join(save_as, "particles.dat"):
                shutil.move(pack_tmp_file, os.path.join(save_as, "particles.dat"))
                pack_tmp_file = os.path.join(save_as, "particles.dat")
        else:
            shutil.move(os.path.join(self.location, str(self.label)+".csv"), os.path.join(save_as, str(self.label)+".csv"))
        self.location = save_as  # update internal file location so we don't lose access to the particle data


class Unit_Conversion:
    """ The base/natural units of rebound are:
    distance: au
    mass: solar_mass
    time: year/2pi (labelled "yr2pi")

    note they are the natural units because with these units, G = 1

    methods unit_from_SI will convert from SI units to the natural unit

    e.g. dist_from_SI(1,000,000) will return 6.68e-6 because is not much of an au

    methods unit_to_SI will convert from the natural unit used by rebound to the SI unit

    e.g. dist_to_SI(0.00001) will return 1495979 because 0.00001 au = 1495979 m

    """
    def __init__(self):
        self.unit_SI = {"m": 1.,
                        "au": 1.495978707e11,
                        "pc": 3.086e16,
                        "s": 1.,
                        "yr": 60*60*24*365.25,
                        "yr2pi": 60*60*24*365.25*(1/(2*np.pi)), #todo: This uses a Julian year, and should be corrected
                        "kg": 1.,
                        "solarmass": 1.98847e30}

    def dist_to_SI(self, x):
        return x*self.unit_SI["au"]

    def dist_from_SI(self, x):
        return x/self.unit_SI["au"]

    def time_to_SI(self, t):
        return t*self.unit_SI["yr2pi"]

    def time_SI_to_yr(self, t):
        return t/self.unit_SI["yr"]

    def time_yr_to_SI(self, t):
        return t*self.unit_SI["yr"]

    def time_from_SI(self, t):
        return t/self.unit_SI["yr2pi"]

    def vel_to_SI(self, v):
        return v * (self.unit_SI["au"]/self.unit_SI["yr2pi"])

    def vel_from_SI(self, v):
        # to covert back we just need to inverse of the previous conversion factor.
        return v * (self.unit_SI["yr2pi"]/self.unit_SI["au"])

    def acc_to_SI(self, a):
        return a * (self.unit_SI["au"]/self.unit_SI["yr2pi"]**2)

    def acc_from_SI(self, a):
        return a * (self.unit_SI["yr2pi"]**2/self.unit_SI["au"])

    def mass_to_SI(self, m):
        return m * self.unit_SI["solarmass"]

    def mass_from_SI(self, m):
        return m / self.unit_SI["solarmass"]

    def dist_au_to_pc(self, x):
        return self.dist_to_SI(x) / self.unit_SI["pc"]

    def dist_SI_to_pc(self, x):
        return x / self.unit_SI["pc"]

    def dist_pc_to_SI(self, x):
        return x * self.unit_SI["pc"]

    def vel_SI_to_pcs(self, v):
        return v * (self.unit_SI["s"]/self.unit_SI["pc"])

class Size_Distribution:
    """ A Size Distribution object stores q values of a size piece-wise defined power law of:
    N(<R) = kR^{-q}

    for example;
    q = 2 between 0 and 5
    q = 3 otherwise

    This object works by storing a 'default' q for the otherwise case, and then added other q's with respective ranges
    via the add_q(q, min, max) method.
    """

    def __init__(self, default_q=3, k=1):
        """

        :param default_q: Sets the 'otherwise' case
        :param k: sets the normalisation constant
        """
        self.uniform = False
        if type(default_q) is str:
            if default_q == "meech-hainaut-marsden":
                default_q = 1.45
            elif default_q == "uniform":
                self.uniform = True
                default_q = 1
            else:
                default_q = float(default_q)
        assert default_q > 0
        self.q_values = [default_q]
        self.q_ranges = [(0, np.inf)]
        self.k = k

    def add_q(self, value, min_size, max_size):
        """

        :param value: q value (must be positive)
        :param min_size: starting size over which this q is valid (inclusive)
        :param max_size: ending size over which this q is valid (exclusive)
        """
        assert max_size > min_size
        assert value > 0  # enforce convention that q is positive and the power law is k^{-q}
        self.q_values.append(value)
        self.q_ranges.append((min_size, max_size))

    def get_uniform(self):
        return self.uniform

    def get_number(self, R):
        """ Returns the number of comets expected exist with a radius less than R.

        :param R: The radius being considered.
        :return: A count of the comets to be expected.
        """
        n = 0
        assert R > 0
        if self.uniform:
            return 1
        else:
            for i in range(len(self.q_values)):
                # because the default value is stored in the first index, the default will be assigned, but then overwritten
                # if another q value exists for the particular radius considered.
                if R < self.q_ranges[i][1] and R >= self.q_ranges[i][0]:
                    n = self.k * R ** (-self.q_values[i])
        return n

    def plot_size_distribution(self, save_as=None):
        """ Creates a plot of the objects internal state.

        :return:
        """
        missed = []
        last_start = 0.1
        # starts from 1 to skip the default case as this is handled separately
        for i in range(1, len(self.q_values)):
            start = self.q_ranges[i][0]
            if start > last_start:
                missed.append([last_start, start])
                last_start = start
            if start == 0:
                start = 0.01
            x = np.linspace(start, self.q_ranges[i][1], 100)
            y = np.zeros(100)
            for j in range(100):
                y[j] = self.get_number(x[j])
            plt.loglog(x, y)
        for i in range(len(missed)):
            x = np.linspace(missed[i][0], missed[i][1], 100)
            y = np.zeros(100)
            for j in range(100):
                y[j] = self.get_number(x[j])
            plt.loglog(x, y)
        plt.grid()
        plt.xlabel("Radius of comet (km)")
        plt.ylabel("Number of comets")
        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as, dpi=600)

    def print_fit(self):
        """
        This prints the internal state of the object. Generally called after the object has been fitted to data,
        because calling it after manually load the data is kinda pointless.

        Also it doesn't print the default, because the default is not used for a fitted object.
        :return:
        """
        for i in range(len(self.q_values)):
            print("found q="+str(self.q_values[1]), "in range ["+str(self.q_ranges[i][0]), str(self.q_ranges[i][1])+")")

    @staticmethod
    def power_relation(x, alpha, k):
        return k * (x ** (-1*alpha))

    def fit_size_distribution(self, radii, range_size=0.5):
        """ This should fit a size distribution to an array of radii. It will create a stepwise function to fit
        the given array.

        The function will clear itself, and then add create itself to be the fitted size distribution. Therefore
        always create a blank size distribution object before calling this function.

        :param radii: Array of radii
        :param range_size: default=0.5, it iterates over the range size, finding a fit for each range size block
        in order to replicate a step-wise function
        :return: Nothing. It sets itself to represent the size distribution the size distribution found will be
        encoding in the object this is called on.
        """
        # Clear the existing object.
        self.__init__()
        low = min(radii)
        high = max(radii) + 1  # this is so we have an extra bin for the largest object
        bins = int(len(radii)*0.2)
        y = np.zeros(bins)
        x = np.linspace(low, high, bins)
        for j in range(len(radii)):
            for i in range(0, bins - 1):
                #  this puts radii  into bins.
                if x[i] <= radii[j] < x[i + 1]:
                    y[i] += 1
                    break  # once we found the right bin we can safely break the inner for loop
        print(y.tolist())
        print(x.tolist())
        s = low
        while s < high:
            index_l = bisect.bisect_left(x, s)
            print("left: "+str(index_l))
            index_r = bisect.bisect_right(x, s+range_size)
            print("right: "+str(index_r))
            print(x[index_l:index_r])
            print(y[index_l:index_r])
            try:
                popt, pcov = optimize.curve_fit(self.power_relation, x[index_l:index_r], y[index_l:index_r], p0=[1, 1])
                fitted_q = popt[0]
                print(popt)
                self.add_q(fitted_q, s, s+range_size)
            except:
                print("Couldn't find fit for range ("+str(s)+", "+str(s+range_size)+"]")
            s += range_size

        self.print_fit()
