'''
        This script builds a simulation with the sun and planets based on nasa horizons data.
        It saves the simulation so that it can be used latter without redownloading data.
'''

import rebound


def build_simulation(save_as="planets.bin", include_inner=False, just_jupiter=False):
    sim = rebound.Simulation()
    sim.add("Sun")
    if just_jupiter:
        sim.add("Jupiter")
    else:
        if include_inner:
            sim.add("Mercury")
            sim.add("Venus")
            sim.add("Earth")
            sim.add("Mars")
        sim.add("Jupiter")
        sim.add("Saturn")
        sim.add("Uranus")
        sim.add("Neptune")
    sim.save(save_as)
