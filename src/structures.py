import numpy as np
import math as mt
import tracing as tr

class VortexParticle:
        def __init__(self, position: np.ndarray, vorticity: int):
            self.position = position
            self.vorticity = vorticity

class FluidSolver:

    def __init__(self, particles: np.ndarray, n_particles: int, boundary: np.ndarray):
        self.particles = particles
        self.n_particles = n_particles
        self.boundary = boundary



