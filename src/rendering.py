import numpy as np
import math as mt
import vortexSolver as vs
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import tracing as tr

max_iterates = 20
frames = []

sim = vs.FluidSolver(vs.initialGrid())
frames.append((sim.positions.copy(), sim.vorticities.copy()))

# tracking code
#center = np.array([3.2, 3.2])
#initial_angles = np.arctan2(sim.positions[:,1] - center[1], sim.positions[:,0] - center[0])

for t in range(1, max_iterates + 1):
    sim.step()
    print(t, "done")
    error = sim.computeError()
    print(error)
    tr.trace(error, "timestep", (sim.time))
    frames.append((sim.positions.copy(), sim.vorticities.copy()))

fig, ax = plt.subplots()
scatter = ax.scatter(frames[0][0][:,0], frames[0][0][:,1], c=frames[0][1], cmap='coolwarm')
#scatter = ax.scatter(frames[0][0][:,0], frames[0][0][:,1], c=initial_angles, cmap='hsv')
plt.colorbar(scatter)

def update(i):
    scatter.set_offsets(frames[i][0])
#    scatter.set_array(frames[i][1])
    ax.set_title(f"t = {i}")
    return [scatter]

animation = ani.FuncAnimation(fig, update, frames=len(frames), interval=150)
animation.save('../VortexParticleSolver/testoutputs/VortexSim.gif', writer='pillow', fps=1)
plt.show()


#print("start of")
#sim.step()
#print("step")
#image = plt.scatter(sim.positions[:,0], sim.positions[:,1], c=sim.vorticities, cmap='coolwarm')
#plt.colorbar(image)
#plt.show()


