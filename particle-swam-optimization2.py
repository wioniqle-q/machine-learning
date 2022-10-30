import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 

def f(x,y):
    "Objective function"
    return np.sin(x) + np.sin((10/3)*x) + np.sin((100/3)*x) + np.sin(y) + np.sin((10/3)*y) + np.sin((100/3)*y)

def swarm_optimization(f, x_min, x_max, y_min, y_max, n_particles, n_iterations, w, c1, c2):
    class Particle:
        def __init__(self):
            self.position = np.random.uniform(x_min, x_max, size=2)
            self.velocity = np.zeros(2)
            self.best_position = self.position
            self.cost = f(*self.position)
            self.best_cost = self.cost

        def update_velocity(self, g_best_position):
            r1 = np.random.random()
            r2 = np.random.random()
            cognitive_velocity = c1 * r1 * (self.best_position - self.position)
            social_velocity = c2 * r2 * (g_best_position - self.position)
            self.velocity = w * self.velocity + cognitive_velocity + social_velocity

        def update_position(self, bounds):
            self.position = self.position + self.velocity
            self.position = np.clip(self.position, bounds[0], bounds[1])

        def update_best_position(self):
            cost = f(*self.position)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_position = self.position

    particles = [Particle() for _ in range(n_particles)]
    g_best_particle = min(particles, key=lambda x: x.best_cost)
    g_best_position = g_best_particle.best_position
    g_best_cost = g_best_particle.best_cost
    history = np.zeros((n_iterations, 2))
    for i in range(n_iterations):
        for particle in particles:
            particle.update_velocity(g_best_position)
            particle.update_position(bounds=[(x_min, x_max), (y_min, y_max)])
            particle.update_best_position()
            if particle.best_cost < g_best_cost:
                g_best_cost = particle.best_cost
                g_best_position = particle.best_position
        history[i] = g_best_position
    return g_best_position, g_best_cost, history

if __name__ == '__main__':
    x, y = np.array(np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100)))
    z = f(x, y)

    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]

    n_particles = 50
    n_iterations = 1000
    w = 0.5
    c1 = 1
    c2 = 2

    g_best_position, g_best_cost, history = swarm_optimization(f, x_min=-10, x_max=10, y_min=-10, y_max=10, n_particles=n_particles, n_iterations=n_iterations, w=w, c1=c1, c2=c2)
    print('Global best position: {}'.format(g_best_position))
    print('Global best cost: {}'.format(g_best_cost))

    fig, ax = plt.subplots()
    ax.contourf(x, y, z, 100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Swarm Optimization')
    ax.plot(history[:, 0], history[:, 1], 'r-')
    ax.plot(history[0, 0], history[0, 1], 'go')
    ax.plot(history[-1, 0], history[-1, 1], 'ro')
    plt.show()
