import numpy as np
import matplotlib.pyplot as plt
import time as time_
from matplotlib.ticker import StrMethodFormatter
import os
import csv


class CustomException(Exception):
    """Allows for a custom exception to be raised.
    """
    pass


class Plotter:
    """Class for various plotting tools.

    Can use either raw data from a simulation, or saved data from a csv.

        Typical usage:

        foo = integrate_simulation()
        p = data.Ploter(foo)
        p.plot_aei(show=True, save_as="Figure1.png", alim=80)

        foo = ["particle1.csv", "particle2.csv"]
        p = data.Ploter(foo)
        p.plot_aei()
    """

    def __init__(self, data):
        """ Constructor for Plotter object.
        Args:
            data: either a list of a, e, i, t lists
            or a list of file names or a file name.
        """
        self.data = []

        if type(data) is str:
            self.load_csv(data)
        elif type(data) is list:
            if type(data[0]) is str:
                self.load_csv(data)
            else:
                self.data = data
        else:
            raise CustomException(str(type(data)) + " is not a valid data type for Plotter.")

    def load_csv(self, data):
        """Loads csv file, generally only called by the constructor.
        Args:
            data: either str of file name of list of str file names.
        """
        self.files = []
        self.a = []
        self.e = []
        self.inc = []
        self.t = []
        if type(data) is str:
            self.files = [data]
        else:
            self.files = data
        for i in range(len(self.files)):
            if not type(self.files[i]) is str:
                raise CustomException(str(type(self.files[i])) + " is not a valid data type for Plotter.")
            self.a.append([])
            self.e.append([])
            self.inc.append([])
            with open(self.files[i], newline="\n") as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                next(reader)  # this skips the header
                for row in reader:
                    self.a[i].append(row[0])
                    self.e[i].append(row[1])
                    self.inc[i].append(row[2])
                    self.t[i].append(row[3])

        print(self.a)


p = Plotter("Sim22-3-2020_19:42/aeit.csv_particle_0")

