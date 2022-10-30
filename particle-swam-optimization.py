import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 

def f(x,y):
    "Objective function"
    return np.sin(x) + np.sin((10/3)*x) + np.sin((100/3)*x) + np.sin(y) + np.sin((10/3)*y) + np.sin((100/3)*y)

class Particle:
    def __init__(self, x, y, v_x, v_y, pbest_x, pbest_y, pbest_cost, cost):
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.pbest_x = pbest_x
        self.pbest_y = pbest_y
        self.pbest_cost = pbest_cost
        self.cost = cost
    
    def __str__(self):
        return f"Particle(x={self.x}, y={self.y}, v_x={self.v_x}, v_y={self.v_y}, pbest_x={self.pbest_x}, pbest_y={self.pbest_y}, pbest_cost={self.pbest_cost}, cost={self.cost})"
    
    def __repr__(self):
        return self.__str__()
    
    def update(self, gbest_x, gbest_y, w, c1, c2):
        self.v_x = w*self.v_x + c1*np.random.random()*(self.pbest_x - self.x) + c2*np.random.random()*(gbest_x - self.x)
        self.v_y = w*self.v_y + c1*np.random.random()*(self.pbest_y - self.y) + c2*np.random.random()*(gbest_y - self.y)

        self.x = self.x + self.v_x
        self.y = self.y + self.v_y

        self.cost = f(self.x, self.y)

        # Update personal best
        if self.cost < self.pbest_cost:
            self.pbest_x = self.x
            self.pbest_y = self.y
            self.pbest_cost = self.cost
        
class Swarm:
    def __init__(self, n_particles, x_min, x_max, y_min, y_max, w, c1, c2):
        self.n_particles = n_particles
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = []
        self.gbest_x = None
        self.gbest_y = None
        self.gbest_cost = np.inf
        self.costs = []
        self.gbest_costs = []
        self.xs = []
        self.ys = []
        self.gbest_xs = []
        self.gbest_ys = []
        self.init_particles()
        
    def init_particles(self):
        for _ in range(self.n_particles):
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            v_x = np.random.uniform(-1, 1)
            v_y = np.random.uniform(-1, 1)
            pbest_x = x
            pbest_y = y
            pbest_cost = f(x, y)
            cost = pbest_cost
            particle = Particle(x, y, v_x, v_y, pbest_x, pbest_y, pbest_cost, cost)
            self.particles.append(particle)

            if particle.cost < self.gbest_cost:
                self.gbest_x = particle.x
                self.gbest_y = particle.y
                self.gbest_cost = particle.cost
            
    def update(self):
        for particle in self.particles:
            particle.update(self.gbest_x, self.gbest_y, self.w, self.c1, self.c2)

            if particle.cost < self.gbest_cost:
                self.gbest_x = particle.x
                self.gbest_y = particle.y
                self.gbest_cost = particle.cost
                
        self.costs.append(self.gbest_cost)
        self.gbest_costs.append(self.gbest_cost)
        self.xs.append(self.gbest_x)
        self.ys.append(self.gbest_y)
        self.gbest_xs.append(self.gbest_x)
        self.gbest_ys.append(self.gbest_y)
        
    def run(self, n_iterations):
        for _ in range(n_iterations):
            self.update()
        
    def plot(self):
        plt.plot(self.costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

    def plot2(self):
        plt.plot(self.xs, self.ys)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        
def main():
    swarm = Swarm(n_particles=100, x_min=-10, x_max=10, y_min=-10, y_max=10, w=0.5, c1=1, c2=2)
    swarm.run(n_iterations=100)
    swarm.plot2()

if __name__ == "__main__":
    main()
