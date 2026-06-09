import numpy as np
import math as mt
import vortexSolver as vs
import matplotlib.pyplot as plt

max_time = 1

sim = vs.FluidSolver(vs.initialGrid())
for t in range(max_time):
    print("start of", t)
    sim.step()
    print("step")
    plt.scatter(sim.positions[:,0], sim.positions[:,1], c=sim.vorticities)
    plt.show()


