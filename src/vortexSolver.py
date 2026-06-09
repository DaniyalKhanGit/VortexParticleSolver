import numpy as np
import math as mt
import tracing as trace

# constants

time_step = 0.5
x0 = 32
y0 = 32
kv = 0.1
t0 = 100
# frame positions from 0 to 63 respectively
frame_size = 64
cirStr = 1
dx = 0.1
smoothing = 1

#
class FluidSolver:
    def __init__(self, particles: tuple, boundary=None):
        self.positions = particles[0]
        self.vorticities = particles[1]
        self.n_particles = np.size(particles[1])
        self.boundary = boundary
        self.velocity_field = np.zeros((frame_size**2, 2))
        self.velocity_prev = np.zeros((frame_size**2, 2))
        self.time = 0

    # scan through vortons
    def prepare_solver(self, new_time: int):
        self.time = new_time

        for i in range(self.n_particles):
            if self.vorticities[i] == -10000:
                # remove the particle
                tempVor = self.vorticities[i]
                tempPos = self.positions[i]
                indice = self.n_particles - 1

                self.vorticities[i] = self.vorticities[indice]
                self.positions[i] = self.positions[indice]
                self.vorticities[indice] = 0
                self.positions[indice] = 0

                self.n_particles = 0

    # also need improvement
    def compute_velocity_field(self):

        self.velocity_prev = self.velocity_field

        # outer loop for each particle in the field
        for i in range(self.n_particles):
            xth_pos = self.positions[i]
            summation = 0
            
            for j in range(self.n_particles):
                if j == i:
                    continue

                iX = xth_pos - self.positions[j]
                iP = np.linalg.norm(iX) / smoothing
                funcQ = (1 / (2 * mt.pi)) * (1 - np.exp((-(iP**2)) / 2))
                circulation = np.array([-self.vorticities[j] * iX[1], self.vorticities[j] * iX[0]])
                sum1 = (circulation * funcQ) / ((np.linalg.norm(iX)**2) + mt.exp(-100))
                summation += sum1
            self.velocity_field[i] = -summation

        return

    # need improve
    def convection(self):
        for i in range(self.n_particles):
            xth = self.positions[i]
            xth_vor = self.vorticities[i]

            if (self.time == 0):
                self.positions[i] += time_step * self.velocity_field[i]
            else:
                self.positions[i] = xth + (time_step * (((3/2) * self.velocity_field[i]) - ((1/2) * self.velocity_prev[i])))

        return

    def diffusion(self):
        self.positions += np.random.uniform(0, mt.sqrt(2 * time_step * 0.01))

    def step(self):
        self.prepare_solver(self.time + time_step)
        print("prep done")
        self.compute_velocity_field()
        print("compute done")
        self.convection()
        print("convect done")
        self.diffusion()
        print("Diffuse done")


# general functions
def initVor(x: np.ndarray) -> int: 
    r_squared = (x[0] - x0)**2 + (x[1] - y0)**2
    return (cirStr / (4 * np.pi * kv * t0)) * np.exp(-r_squared / (4 * kv * t0))

def initialGrid() -> tuple:

    positions = np.zeros((frame_size**2, 2))
    vorticities = np.zeros(frame_size**2)
    
    for x in range(frame_size):
        for y in range(frame_size):
            positionTemp = np.array([x, y])
            posVorticity = initVor(positionTemp)

            index = x * frame_size + y
            positions[index] = np.array([x * dx, y * dx])
            vorticities[index] = posVorticity
    
    return (positions, vorticities)

