import numpy as np
import math as mt
import vortexSolver as vs
import matplotlib.pyplot as plt
import matplotlib.animation as ani

max_iterates = 5
frames = []

sim = vs.FluidSolver(vs.initialGrid())
for t in range(max_iterates):
    sim.step()
    print(t, "done")
    frames.append((sim.positions.copy(), sim.vorticities.copy()))

fig, ax = plt.subplots()
scatter = ax.scatter(frames[0][0][:,0], frames[0][0][:,1], c=frames[0][1], cmap='coolwarm')
plt.colorbar(scatter)

def update(i):
    scatter.set_offsets(frames[i][0])
    scatter.set_array(frames[i][1])
    ax.set_title(f"t = {i}")
    return [scatter]

animation = ani.FuncAnimation(fig, update, frames=len(frames), interval=1000)
animation.save('../VortexParticleSolver/testoutputs/VortexSim.gif', writer='pillow', fps=1)
plt.show()


#print("start of")
#sim.step()
#print("step")
#image = plt.scatter(sim.positions[:,0], sim.positions[:,1], c=sim.vorticities, cmap='coolwarm')
#plt.colorbar(image)
#plt.show()


