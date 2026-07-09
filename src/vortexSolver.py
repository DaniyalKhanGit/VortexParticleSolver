import numpy as np
import math as mt
import tracing as trace


# constants

time_step = 1
dx = 0.1
x0 = 32 * dx
y0 = 32 * dx
kv = 0.1
t0 = 0
epsilon = np.exp(-10)
diffusionScaling = 0.2

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

        self.b_lengths = None
        self.b_midpoints = None
        self.b_normals = None
        self.b_tangents = None
        self.startpoints = None
    
        self.rotationMatrix = None

        if boundary != None:
            self.boundary_status = True

            computed = boundaryComputation(self.boundary)

            self.b_lengths = computed[0]
            self.b_midpoints = computed[1]
            self.b_tangents = computed[2]
            self.b_normal = computed[3]
            self.startpoints = computed[4]
            self.rotationMatrix = computed[5]

            # boundary matrix handling here

    # scan through vortons
    def prepare_solver(self, new_time: int):
        self.time = new_time
        
        '''
        for i in range(self.n_particles):
            if self.vorticities[i] == 0:
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
        self.positions += (np.random.normal(0, np.sqrt(2 * time_step * 0.01), size=self.positions.shape) * diffusionScaling)

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
'''
def initVor(x: np.ndarray) -> int: 
    r_squared = (x[0] - x0)**2 + (x[1] - y0)**2
    return (cirStr / (4 * np.pi * kv * t0)) * np.exp(-r_squared / (4 * kv * t0))
'''

def initVor(x: np.ndarray) -> int:
    cx = 32
    cy = 32
    r = 10 * dx

    value = 0
    value += np.exp(-((x[0] - (x0-r))**2 + (x[1]-(y0+r))**2) / (50 * dx * dx))
    value -= np.exp(-((x[0] - (x0+r))**2 + (x[1]-(y0+r))**2) / (50 * dx * dx))
    value -= np.exp(-((x[0] - (x0-r))**2 + (x[1]-(y0-r))**2) / (50 * dx * dx))
    value += np.exp(-((x[0] - (x0+r))**2 + (x[1]-(y0-r))**2) / (50 * dx * dx))
    return value

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

def boundaryComputation(boundary) -> tuple:
    # [x, y], [a, b]
    startpoints = np.array([b[0] for b in boundary])
    endpoints = np.array([b[1] for b in boundary])
    diff = endpoints - startpoints

    length = np.linalg.norm(diff, axis=1)
    midpoint = (startpoints + endpoints) / 2
    tangents = diff / length[:, None]
    normal = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    rotation_matrix = np.stack([tangents, normal], axis=1)

    return (length, midpoint, tangents, normal, startpoints, rotation_matrix)


def boundaryMatrixConstruction(boundary: np.ndarray, midpoints, lengths, rotationMs, starts):
    
    # for this we gotta first translate to local coords, we do this vectorized since this is N^2
    # then do our little comps for u* and v*, and then translate back
    # then we just dot the normal, and we got our entries for the matrix A

    delta = midpoints[None, :, :] - starts[:, None, :]
    local_coords = np.einsum("ikl,ijl->ijk", rotationMs, delta)

