
from NeuralNetworks import *
from random import randint
from joblib import Parallel, delayed
import numpy as np
import copy
from game import *
import multiprocessing

class genetic_Algorithm():
    
    def __init__(self, neuralNets=None, neuralNetShape = None,
                 populationSize = 1000, generationSize=100, mutationRate=0.7, crossoverRate=0.3, 
                 crossoverType="Neuron_change", mutationType="Weight_change"):
        self.neuralNetShape = neuralNetShape
        if self.neuralNetShape==None:
            self.neuralNetShape = [21, 16, 3]
            # 21 imputs, 18 hiddenLayer count, 3 outputLayer count
        
        self.neuralNets = neuralNets
        if self.neuralNets == None: # create random population sized neuralNets.
            self.neuralNets = []
            for i in range(populationSize):
                self.neuralNets.append(NeuralNet(self.neuralNetShape))
        
        self.populationSize = populationSize
        self.generationSize = generationSize
        self.mutationRate = mutationRate
        self.mutationType = mutationType
        self.crossoverRate = crossoverRate
        self.crossoverType = crossoverType
    
    def start_generation(self):
        
        neuralNets = self.neuralNets
        populationSize = self.populationSize
        mutationRate = self.mutationRate
        crossoverRate = self.crossoverRate
        
        crossover_count = int(crossoverRate * populationSize)
        mutation_count = int(mutationRate * populationSize)
        
        cpu_cores_count = multiprocessing.cpu_count()
        gen  = 0
        for i in range(self.generationSize):
            gen += 1
            parentsSelected = self.parentSelect(neuralNets, crossover_count, populationSize)  # 300
            childrenProduces = self.childrenProduction(parentsSelected, crossover_count)    # 300
            mutations = self.mutationProduction(neuralNets, populationSize, mutation_count) # 700
            
            neuralNets = neuralNets + childrenProduces + mutations # 1000 + 300 + 700 = 2000
            self.filter(neuralNets, cpu_cores_count) # filter will run all networks parallely to find their scores.
            neuralNets.sort(key=lambda neuralNet : neuralNet.score, reverse=True)
            neuralNets[0].save(name = "generation_"+str(gen))
        
        for i in range(int(0.2*len(neuralNets))): # further mutation for best results.
            randInd = randint(10, len(neuralNets)-1) 
            neuralNets[randInd] = self.mutate(neuralNets[randInd])
        
        neuralNets = neuralNets[:populationSize] # trimming best 
        self.Display_GenDetails(self.neuralNets, gen)
    
    
    def parentSelect(self, neuralNets, crossover_cnt, populationSize):
        parents_selected = []
        for i in range(crossover_cnt):
            par = self.tournament(neuralNets[randint(0, populationSize-1)],
                                  neuralNets[randint(0, populationSize-1)],
                                  neuralNets[randint(0, populationSize-1)] )
            parents_selected.append(par)
        return parents_selected
    
    def childrenProduction(self, parents, crossover_cnt):
        children_selected = []
        for i in range(crossover_cnt):
            childNet = self.crossover(parents[randint(0, crossover_cnt-1)],
                                      parents[randint(0, crossover_cnt-1)])
            children_selected.append(childNet)
        return children_selected
        
    def mutationProduction(self, neuralNets, populationSize, mutation_count):
        mutations_produced = []
        for i in range(mutation_count):
            mutation = self.mutate( neuralNets[randint(0, populationSize-1)])
            mutations_produced.append(mutation)
        return mutations_produced
    
    def tournament(self, par1, par2, par3):
        # par1, par2, par3 are 3 neuralnets
        game = Game()
        game.start(neural_net = par1)
        score1 = game.game_score
        game.start(neural_net = par2)
        score2 = game.game_score
        game.start(neural_net = par3)
        score3 = game.game_score
        mx = max(score1, score2, score3)
        if mx == score1:
            return par1
        elif mx == score2:
            return par2
        return par3
    
    def filter(self, neuralNets, cores_cnt):
        game = Game()
        results_1 = Parallel(n_jobs=cores_cnt)(delayed(game.start)(neural_net=neuralNets[i]) for i in range(len(neuralNets)))
        results_2 = Parallel(n_jobs=cores_cnt)(delayed(game.start)(neural_net=neuralNets[i]) for i in range(len(neuralNets)))
        results_3 = Parallel(n_jobs=cores_cnt)(delayed(game.start)(neural_net=neuralNets[i]) for i in range(len(neuralNets)))
        results_4 = Parallel(n_jobs=cores_cnt)(delayed(game.start)(neural_net=neuralNets[i]) for i in range(len(neuralNets)))
        for i in range(len(results_1)):
            neuralNets[i].score = int(np.mean([results_1[i],
                                               results_2[i],
                                               results_3[i],
                                               results_4[i]]) )
    
    def crossover(self, net1, net2):
        cpy1 = copy.deepcopy(net1)
        cpy2 = copy.deepcopy(net2)
        weight_or_bias = randint(0,1)
        if weight_or_bias==0: # over weights or neurons or layers
            if self.crossoverType=="Weight_change":
                layer = randint(0, len(cpy1.weights) - 1)
                neuron = randint(0, len(cpy1.weights[layer]-1))
                weight = randint(0, len(cpy1.weights[layer][neuron]-1))
                temp = cpy1.weights[layer][neuron][weight]
                cpy1.weights[layer][neuron][weight] = cpy2.weights[layer][neuron][weight]
                cpy2.weights[layer][neuron][weight] = temp
            elif self.crossoverType =="Neuron_change":
                layer = randint(0, len(cpy1.weights) - 1)
                neuron = randint(0, len(cpy1.weights[layer])-1)
                temp = copy.deepcopy(cpy1)
                cpy1.weights[layer][neuron] = cpy2.weights[layer][neuron]
                cpy2.weights[layer][neuron] = temp.weights[layer][neuron]
            elif self.crossoverType == "Layer_change":
                layer = randint(0, len(cpy1.weights) - 1)
                temp  = copy.deepcopy(cpy1)
                cpy1.weights[layer] = cpy2.weights[layer]
                cpy2.weights[layer] = temp.weights[layer]
        else :
            layer = randint(0, len(cpy1.biases)-1)
            bias = randint(0, len(cpy1.biases[layer])-1)
            temp = copy.deepcopy(cpy1)
            cpy1.weights[layer][bias] = cpy2.weights[layer][bias]
            cpy2.weights[layer][bias] = temp.weights[layer][bias]
            
        game = Game()
        game.start(neural_net=cpy1)
        scr1 = game.game_score
        game.start(neural_net=cpy2)
        scr2 = game.game_score
        
        if(scr1 > scr2):
            return cpy1
        return cpy2
    
    def mutate(self, neuralNet):
        
        cpy1 = copy.deepcopy(neuralNet)
        weight_or_bias = randint(0, 1)
        
        if weight_or_bias==0: # over weights or neurons or layers
            if self.crossoverType=="Weight_change":
                layer = randint(0, len(cpy1.weights) - 1)
                neuron = randint(0, len(cpy1.weights[layer]-1))
                weight = randint(0, len(cpy1.weights[layer][neuron]-1))
                cpy1.weights[layer][neuron][weight] = np.random.randn()
            elif self.crossoverType =="Neuron_change":
                layer = randint(0, len(cpy1.weights) - 1)
                neuron = randint(0, len(cpy1.weights[layer])-1)
                cpy1.weights[layer][neuron] = np.random.randn(len(cpy1.weights[layer][neuron]))
        
        else :
            layer = randint(0, len(cpy1.biases)-1)
            bias = randint(0, len(cpy1.biases[layer])-1)
            cpy1.weights[layer][bias] = np.random.randn()
        
        return cpy1
    
    
    def Display_GenDetails(self, neuralNets, gen):
        
        top_mean_score = int(np.mean([neuralNets[i].score for i in range(6)]))
        bottom_mean_score = int(np.mean([neuralNets[-i].score for i in range(1,6)]))
        print("\nbest fitness in generation ", gen , " : ", neuralNets[0].score)
        print("Population size = ", len(neuralNets))
        print("top 6 avg score = ", top_mean_score)
        print("bottom 6 avg score = ", bottom_mean_score)
        
        