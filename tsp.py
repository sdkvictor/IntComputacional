from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np


class TravelingSalesman:

    def __init__(self, name):
        self.distances = []
        self.initData()

    def __len__(self):
        return len(self.distances)

    def initData(self):

        self.distances = [  [0, 10, 15, 20], 
                            [10, 0, 35, 25], 
                            [15, 35, 0, 30], 
                            [20, 25, 30, 0] ]

    def getDistance(self, indices):
        totalDistance = 0
        #indices = [1,3,4,2]
        for i in range(len(indices-1)):
            totalDistance += self.distances[indices[i]][indices[i+1]]
            
        return totalDistance


    def printResult(self, indices):
        print("Best route = ", indices)
        print("Total distance = ", getDistance(indices))


#create instance of knapsack problem class
tsp = TravelingSalesman()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# evaluation function
def func_eval(individual):
    return tsp.getDistance(individual),  


toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))
toolbox.register("evaluate", func_eval)

toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.randomOrder)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main():

    pop = toolbox.population(n=100)

    #statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("min", numpy.min)
    stats.register("mean", numpy.mean)
    stats.register("std", numpy.std)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(1)

    #eaSimple
    print("------- EaSimple -------")

    df = pd.DataFrame()
    for i in range (10):
        pop, log = algorithms.eaSimple(pop, toolbox, 1.0, 0.5, 100, stats=stats, halloffame=hof)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = 'eaSimple'
        df2['corrida'] = i
        df = df.append(df2)

    df = df.reset_index(drop=True)

    for i in range(1010):
        if i > 0 and df.at[i, 'max'] < df.at[i-1, 'max']:
            df.at[i, 'max'] = df.at[i-1, 'max']

    print(df.to_string())

    # print best solution found:
    best = hof.items[0]
    print("Best Ever Individual = ", best)
    print("Best Ever Fitness = ", best.fitness.values[0])

    print("--- TSP best route --- ")
    tsp.printItems(best)

    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'max':['mean', 'std']})
    print(df_promedios.to_string())

    x = df['gen'].unique()

    promedios = df_promedios['max']['mean'].values
    desviacion = df_promedios['max']['std'].values
    plt.plot(x, promedios, color='r')
    plt.plot(x, promedios - desviacion, linestyle = '--', color='b')
    plt.plot(x, promedios + desviacion, linestyle = '--', color='g')

    plt.show()

    #--------------------------------------------------------------------------------------------
    
    #eaMuPlusLambda
    print("------- EaMuPlusLambda -------")

    hof2 = tools.HallOfFame(1)
    df = pd.DataFrame()
    for i in range (10):
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 5, 10, 0.5, 0.5, 100, stats=stats, halloffame=hof2)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = 'eaMuPlusLambda'
        df2['corrida'] = i
        df = df.append(df2)

    df = df.reset_index(drop=True)

    for i in range(1010):
        if i > 0 and df.at[i, 'max'] < df.at[i-1, 'max']:
            df.at[i, 'max'] = df.at[i-1, 'max']

    print(df.to_string())

    # print best solution found:
    best = hof2.items[0]
    print("Best Ever Individual = ", best)
    print("Best Ever Fitness = ", best.fitness.values[0])

    print("--- TSP best route --- ")
    knapsack.printItems(best)

    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'max':['mean', 'std']})
    print(df_promedios.to_string())

    x = df['gen'].unique()

    promedios = df_promedios['max']['mean'].values
    desviacion = df_promedios['max']['std'].values
    plt.plot(x, promedios, color='r')
    plt.plot(x, promedios - desviacion, linestyle = '--', color='b')
    plt.plot(x, promedios + desviacion, linestyle = '--', color='g')

    plt.show()

    #--------------------------------------------------------------------------------------------

    #eaMuCommaLambda

    print("------- EaMuCommaLambda -------")
    hof3 = tools.HallOfFame(1)
    df = pd.DataFrame()
    for i in range (10):
        pop, log = algorithms.eaMuCommaLambda(pop, toolbox, 5, 10, 0.5, 0.5, 100, stats=stats, halloffame=hof3)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = 'eaMuCommaLambda'
        df2['corrida'] = i
        df = df.append(df2)

    df = df.reset_index(drop=True)

    for i in range(1010):
        if i > 0 and df.at[i, 'max'] < df.at[i-1, 'max']:
            df.at[i, 'max'] = df.at[i-1, 'max']

    
    print(df.to_string())

    # print best solution found:
    best = hof3.items[0]
    print("Best Ever Individual = ", best)
    print("Best Ever Fitness = ", best.fitness.values[0])

    print("--- TSP best route --- ")
    knapsack.printItems(best)


    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'max':['mean', 'std']})
    print(df_promedios.to_string())

    x = df['gen'].unique()

    promedios = df_promedios['max']['mean'].values
    desviacion = df_promedios['max']['std'].values
    plt.plot(x, promedios, color='r')
    plt.plot(x, promedios - desviacion, linestyle = '--', color='b')
    plt.plot(x, promedios + desviacion, linestyle = '--', color='g')

    plt.show()

if __name__ == "__main__":
    main()


    