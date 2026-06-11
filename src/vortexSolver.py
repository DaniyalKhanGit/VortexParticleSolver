import numpy as np
import math as mt
import tracing as trace


# constants

time_step = 0.1
dx = 0.1
x0 = 32 * dx
y0 = 32 * dx
kv = 0.1
t0 = 1

# frame positions from 0 to 63 respectively
frame_size = 64
cirStr = 10
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
        self.time = t0
        self.boundary_status = False
        self.boundary_matrix = None

        if boundary != None:
            self.boundary_status = True

            # boundary matrix handling here

    # scan through vortons
    def prepare_solver(self, new_time: int):
        self.time = new_time
        '''
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
        '''

    # also need improvement
    def compute_velocity_field(self):

        self.velocity_prev = self.velocity_field.copy()

        # outer loop for each particle in the field
        for i in range(self.n_particles):
            xth_pos = self.positions[i]
            summation = 0

            # vectorized inner loop

            iX = xth_pos - self.positions
            iP = np.linalg.norm(iX, axis=1) / smoothing
            funcQ = (1 / (2 * np.pi)) * (1 - np.exp((-(iP**2)) / 2))
            gamma = self.vorticities * dx**2
            circulation = np.array([-gamma * iX[:,1], gamma * iX[:,0]])
            sum1 = (circulation * funcQ) / ((np.linalg.norm(iX, axis=1)**2) + mt.exp(-100))
            summation = np.sum(sum1, axis=1)

            self.velocity_field[i] = -summation

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
        self.positions += np.random.normal(0, np.sqrt(2 * time_step * 0.01), size=self.positions.shape)

    def force_addition(self):
        pass

    def no_slip_boundary(self):
        pass

    def no_through_boundary(self):
        pass

    def step(self):
        self.prepare_solver(self.time + time_step)
        print("prep done")
        self.compute_velocity_field()
        print("compute done")
        self.convection()
        print("convect done")
        self.diffusion()
        print("Diffuse done")
    
    def computeError(self):
        error = np.sqrt(np.mean((self.vorticities - analyticalVorticity(self.positions, self.time))**2))
        return error
            


# general functions
def initVor(x: np.ndarray) -> int: 
    r_squared = (x[0] - x0)**2 + (x[1] - y0)**2
    return (cirStr / (4 * np.pi * kv * t0)) * np.exp(-r_squared / (4 * kv * t0))

def initialGrid() -> tuple:

    positions = np.zeros((frame_size**2, 2))
    vorticities = np.zeros(frame_size**2)
    
    for x in range(frame_size):
        for y in range(frame_size):
            positionTemp = np.array([x * dx, y * dx])
            posVorticity = initVor(positionTemp)

            index = x * frame_size + y
            positions[index] = np.array([x * dx, y * dx])
            vorticities[index] = posVorticity
    
    return (positions, vorticities)

def analyticalVorticity(positions: np.ndarray, time: int):
    r_sq = (positions[:, 0] - x0)**2 + (positions[:,1] - y0)**2
    return (cirStr / (4 * np.pi * kv * time)) * np.exp(-r_sq / (4 * kv * time))

def boundaryComputation(positionA, positionB) -> tuple:
    # [x, y], [a, b]
    length = np.linalg.norm(positionA - positionB)
    midpoint = np.array([((positionA[0] + positionB[0]) / 2), ((positionA[1] + positionB[1]) / 2)])
    normal = (positionA[:, 0] * positionB[:, 1]) - (positionA[:, 1] * positionB[:,0])

    return (length, midpoint, normal)


def boundaryMatrixConstruction(boundary: np.ndarray):
    pass