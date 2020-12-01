import rebound
import matplotlib.pyplot as plt
import numpy as np
import datetime
import generate

sim = rebound.Simulation()

# Settings
sim.G = 1                   # Sets gravitational constant (default 1)
sim.softening = 1           # Sets gravitational softening parameter (default 0)
sim.testparticle_type = 1   # Allows massive particles to feel influence from testparticles (default 0)
sim.dt = 0.1                # Sets the timestep (will change for adaptive integrators such as IAS15).
sim.t = 0.                  # Sets the current simulation time (default 0)

sim.add(m=5.)

sim = generate.gen(sim)

def heartbeat(sim):
    print(sim.contents.t)

sim.heartbeat = heartbeat
sim.integrate(5)

fig = rebound.OrbitPlot(sim, unitlabel="[AU]")
plt.show()