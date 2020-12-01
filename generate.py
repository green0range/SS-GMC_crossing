import rebound
import numpy as np
def gen(sim, n=100, m=1., d=1000):
    masses = (m+np.random.randn(n)).tolist()
    x_ = (d+np.random.randn(n)).tolist()
    for i in range(0, len(masses)):
        sim.add(m=masses[i], x=x_[i], vy=5.)
    print("added")
    return sim