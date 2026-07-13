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
        self.boundary_velocity_field = np.zeros((frame_size**2, 2))
        self.velocity_prev = np.zeros((frame_size**2, 2))
        self.time = t0
        self.boundary_status = False
        self.boundary_matrix = None
        self.A_debug = None
        self.b_midpoints_velocity = None

        self.b_lengths = None
        self.b_midpoints = None
        self.b_normals = None
        self.b_tangents = None
        self.startpoints = None
    
        self.rotationMatrix = None

        if boundary is not None:
            self.boundary_status = True

            computed = boundaryComputation(self.boundary)

            self.b_lengths = computed[0]
            self.b_midpoints = computed[1]
            self.b_tangents = computed[2]
            self.b_normals = computed[3]
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

            self.velocity_field[i] = summation

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
        # Depreciated (might or might not implement, in the actual model we won't need this constraint but if I wanna test here)
        pass

    def no_through_boundary(self):

        # could also add check later if boundary has changed, but its pretty useless rn
        panelStrengths = self.panelStrengthCalc()
        localEvalCoords = self.localSpaceComputation(self.positions)
        u_p = np.einsum("ijk, i->ik", localEvalCoords, panelStrengths)
        self.boundary_velocity_field = u_p
        self.velocity_field += u_p


    def evalAtMidpoints(self):

        for i in range(len(self.b_midpoints)):
            xth_pos = self.b_midpoints[i]
            summation = 0

            # vectorized inner loop

            iX = xth_pos - self.positions
            iP = np.linalg.norm(iX, axis=1) / smoothing
            funcQ = (1 / (2 * np.pi)) * (1 - np.exp((-(iP**2)) / 2))
            gamma = self.vorticities * dx**2
            circulation = np.array([-gamma * iX[:,1], gamma * iX[:,0]])
            sum1 = (circulation * funcQ) / ((np.linalg.norm(iX, axis=1)**2) + mt.exp(-100))
            summation = np.sum(sum1, axis=1)

            self.b_midpoints_velocity[i] = summation
            # double check after if summation signs correct

    def panelStrengthCalc(self):

        self.evalAtMidpoints()
        u_in = -np.array(self.b_midpoints_velocity.copy()) # we dont got free flow yet so just noting
        vectorB = np.einsum("jk,jk->j", u_in, self.b_normals)
        panelStrs = self.boundary_matrix @ np.append(vectorB, 0.0)
        return panelStrs


    def step(self):
        self.prepare_solver(self.time + time_step)
        print("prep done")
        self.compute_velocity_field()
        print("compute done")
        self.convection()
        print("convect done")
        self.diffusion()
        print("Diffuse done")
        if self.boundary_status:
            self.no_through_boundary()
            print("boundary done")
    
    def computeError(self):
        error = np.sqrt(np.mean((self.vorticities - analyticalVorticity(self.positions, self.time))**2))
        return error

    def localSpaceComputation(self, points):

        delta = points[None, :, :] - self.startpoints[:, None, :]
        local_coords = np.einsum("ikl,ijl->ijk", self.rotationMatrix, delta)

        xlocal = local_coords[..., 0]
        ylocal = local_coords[..., 1]

        r1 = np.sqrt(xlocal**2 + ylocal**2)
        r1plus = np.sqrt((xlocal - self.b_lengths[:, None])**2 + ylocal**2)

        # now time for angle 

        beta = np.arctan2(-ylocal, self.b_lengths[:, None] - xlocal) - np.arctan2(-ylocal, -xlocal)
        beta = np.arctan2(np.sin(beta), np.cos(beta))

        ulocal = beta / (2*np.pi)
        vlocal = (1 / (2*np.pi)) * np.log((r1plus + np.exp(-20)) / (r1 + np.exp(-20)))

        local_velocity = np.stack([ulocal, vlocal], axis=-1)
        return np.einsum("ilk,ijl->ijk", self.rotationMatrix, local_velocity)
    
    def buildBoundaryMatrix(self):

       panelinfluence = self.localSpaceComputation(self.b_midpoints)
       A = np.einsum("ijk,jk->ji", panelinfluence, self.b_normals)
       A_augmented = np.vstack([A, np.ones(A.shape[1])])
       self.boundary_matrix = np.linalg.pinv(A_augmented)
       self.A_debug = A

    

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


def makeSquarePanels(low: np.ndarray, high: np.ndarray, npanels: int) -> np.ndarray:

    corners = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]], dtype=np.float64)

    segments = []
    for c in range(4):
        i1 = corners[c]
        i2 = corners[(c + 1) % 4]
        nodes = np.linspace(i1, i2, npanels + 1)
        for p in range(npanels):
            segments.append([nodes[p], nodes[p + 1]])
    return np.array(segments)


