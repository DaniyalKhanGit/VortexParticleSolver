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
dx = 0.1
smoothing = 1

#
class FluidSolver:
    def __init__(self, particles: np.ndarray, n_particles: int, boundary: np.ndarray, time: int):
        self.positions = particles[0]
        self.vorticities = particles[1]
        self.n_particles = n_particles
        self.boundary = np.size(particles)
        self.velocity_field = np.zeros((frame_size**2, 2))
        self.velocity_prev = np.zeros((frame_size**2, 2))
        self.time = time

    # scan through vortons
    def prepare_solver():
        pass

    def compute_velocity_field():

        FluidSolver.velocity_prev = FluidSolver.velocity_field

        # outer loop for each particle in the field
        for i in range(FluidSolver.n_particles):
            xth_pos = FluidSolver.positions[i]
            summation = 0
            
            for j in range(FluidSolver.n_particles):
                if j == i:
                    continue

                iX = xth_pos - FluidSolver.positions[j]
                iP = abs(iX) / smoothing
                funcQ = (1 / (2 * mt.pi)) * (1 - mt.exp((-(iP**2)) / 2))
                circulation = np.cross(iX, FluidSolver.vorticities[j])
                sum1 = (circulation * funcQ) / (abs(iX)**2)
                summation += sum1
            FluidSolver.velocity_field[i] = -summation

        return

    def convection(time):
        
        for i in range(FluidSolver.n_particles):
            xth = FluidSolver.positions[i]
            xth_vor = FluidSolver.vorticities[i]

            if (time == 0):
                FluidSolver.positions[i] += time_step * FluidSolver.velocity_field[i]
            else:
                FluidSolver.positions[i] = xth + (time_step * (((3/2) * FluidSolver.velocity_field[i]) - ((1/2) * FluidSolver.velocity_prev[i])))

        return

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
            positions[index] = np.array([x * dx, y * dx])
            vorticities[index] = posVorticity
    
    return (positions, vorticities)

