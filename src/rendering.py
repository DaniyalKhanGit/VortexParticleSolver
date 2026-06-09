import numpy as np
import math as mt
import vortexSolver as vs
import matplotlib as plt

max_time = 1

sim = vs.FluidSolver(vs.initialGrid())
for t in range(max_time):
    vs.FluidSolver.step()
    plt.plot(vs.FluidSolver.positions, vs.FluidSolver.vorticities)
    plt.show()


