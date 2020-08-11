
import numpy as np
from numba import jit
import pygame
from cnst import *

class NeuralNet():
    def __init__(self, neuralNetShape=None):
        self.shape = neuralNetShape
        self.weights = []
        self.biases = []
        self.score = 0
        # rand init of biases and weights..
        if neuralNetShape:
            for j in neuralNetShape[1:]:
                self.biases.append(np.random.randn(j,1))
            for i,j in zip(neuralNetShape[:-1], neuralNetShape[1:]):
                self.weights.append(np.random.randn(j,i))
            
        
    def Feed_F(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a)+bias)
        return a
    
    def load(self, weights_filename, biases_filename):
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        self.weights = np.load(weights_filename)
        self.biases = np.load(biases_filename)
        np.load = np_load_old
    
    def save(self, name=None):
        if not name:
            np.save('saved_weights_'+str(self.score), self.weights)
            np.save('saved_biases_'+str(self.score), self.biases)
        else :
            np.save(name + '_weights', self.weights)
            np.save(name + '_bises', self.biases)
            
    def render(self, window, vision):
        # it will display the current state of neuralNet in right side..
        
        # network = [np.array(vision)]            # will contain all neuron activation from each layer
        # for i in range(len(self.biases)):
        #     activation = sigmoid(np.dot(self.weights[i], network[i]) + self.biases[i])  # compute neurons activations
        #     network.append(activation)                                                  # append it

        # screen_division = WINDOW_SIZE / (len(network) * 2)     # compute distance between layers knowing window's size
        # step = 1
        # for i in range(len(network)):                                           # for each layer
        #     for j in range(len(network[i])):                                    # for each neuron in current layer
        #         y = int(WINDOW_SIZE/2 + (j*24) - (len(network[i])-1)/2 * 24)    # neuron position
        #         x = int(WINDOW_SIZE + screen_division * step)
        #         intensity = int(network[i][j][0] * 255)                         # neuron intensity

        #         if i < len(network)-1:
        #             for k in range(len(network[i+1])):                                          # connections
        #                 y2 = int(WINDOW_SIZE/2 + (k * 24) - (len(network[i+1]) - 1) / 2 * 24)   # connections target position
        #                 x2 = int(WINDOW_SIZE + screen_division * (step+2))
        #                 pygame.gfxdraw.line(window, x, y, x2, y2,                               # draw connection
        #                                     (intensity/2+30, intensity/2+30, intensity/2+30, intensity/2+30))

        #         pygame.gfxdraw.filled_circle(window, x, y, 9, (intensity, intensity, intensity))    # draw neuron
        #         pygame.gfxdraw.aacircle(window, x, y, 9, (205, 205, 205))
        #     step += 2
        pass

@jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))