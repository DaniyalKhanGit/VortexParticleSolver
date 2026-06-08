import numpy as np
import math as mt
import tracing as trace

# constants

time_step = 1
x0 = 32
y0 = 32
# frame positions from 0 to 63 respectively
frame_size = 64
cirStr = 1

#
class FluidSolver:
    def __init__(self, particles: np.ndarray, n_particles: int, boundary: np.ndarray):
        self.positions = particles[0]
        self.vorticities = particles[1]
        self.n_particles = n_particles
        self.boundary = np.size(particles)

    def compute_velocity_field():
        pass

    def convection():
        pass

    def diffusion():
        pass

# general functions
def initVor(x: np.ndarray) -> int: 
    r_squared = (x[0] - x0)**2 + (x[1] - y0)**2
    return (cirStr / (4 * np.pi * 0.0001)) * np.exp(-r_squared / (4 * 0.0001))

def initialGrid() -> tuple:

    positions = np.zeros((frame_size**2, 2))
    vorticities = np.zeros(frame_size**2)
    
    for x in range(frame_size):
        for y in range(frame_size):
            positionTemp = np.array(x, y)
            posVorticity = initVor(positionTemp)

            index = x * frame_size + y
            positions[index] = np.array([x, y])
            vorticities[index] = posVorticity
    
    return (positions, vorticities)

