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

    def __init__(self):
        self.distances = []
        self.initData()

    def __len__(self):
        return len(self.distances)

    def initData(self):

        #self.distances = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
        self.distances = [
            [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145],
            [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357],
            [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453],
            [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280],
            [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586],
            [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887],
            [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114],
            [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300],
            [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653],
            [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272],
            [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017],
            [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0],     
        ]

    def getDistance(self, indices):
        totalDistance = 0
        #indices = [1,3,4,2]
        for i in range(len(indices)-1):
            totalDistance += self.distances[indices[i]][indices[i+1]]
        
        totalDistance += self.distances[indices[len(indices)-1]][indices[0]]
            
        return totalDistance


    def printResult(self, indices):
        print("Best route = ", indices)
        totalDistance = 0
        #indices = [1,3,4,2]
        for i in range(len(indices)-1):
            totalDistance += self.distances[indices[i]][indices[i+1]]
        
        totalDistance += self.distances[indices[len(indices)-1]][indices[0]]
        print("Total distance = ", totalDistance)


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
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.randomOrder)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main():

    pop = toolbox.population(n=10)

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
        pop, log = algorithms.eaSimple(pop, toolbox, 1.0, 0.5, 1000, stats=stats, halloffame=hof)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = 'eaSimple'
        df2['corrida'] = i
        df = df.append(df2)

    df = df.reset_index(drop=True)

    for i in range(10010):
        if i > 0 and df.at[i, 'min'] > df.at[i-1, 'min']:
            df.at[i, 'min'] = df.at[i-1, 'min']

    print(df.to_string())

    # print best solution found:
    best = hof.items[0]
    print("Best Ever Individual = ", best)
    print("Best Ever Fitness = ", best.fitness.values[0])

    print("--- TSP best route --- ")
    tsp.printResult(best)

    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'min':['mean', 'std']})
    print(df_promedios.to_string())

    x = df['gen'].unique()

    promedios = df_promedios['min']['mean'].values
    desviacion = df_promedios['min']['std'].values
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
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 5, 10, 0.5, 0.5, 1000, stats=stats, halloffame=hof2)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = 'eaMuPlusLambda'
        df2['corrida'] = i
        df = df.append(df2)

    df = df.reset_index(drop=True)

    for i in range(10010):
        if i > 0 and df.at[i, 'min'] > df.at[i-1, 'min']:
            df.at[i, 'min'] = df.at[i-1, 'min']

    print(df.to_string())

    # print best solution found:
    best = hof2.items[0]
    print("Best Ever Individual = ", best)
    print("Best Ever Fitness = ", best.fitness.values[0])

    print("--- TSP best route --- ")
    tsp.printResult(best)

    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'min':['mean', 'std']})
    print(df_promedios.to_string())

    x = df['gen'].unique()

    promedios = df_promedios['min']['mean'].values
    desviacion = df_promedios['min']['std'].values
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
        pop, log = algorithms.eaMuCommaLambda(pop, toolbox, 5, 10, 0.5, 0.5, 1000, stats=stats, halloffame=hof3)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = 'eaMuCommaLambda'
        df2['corrida'] = i
        df = df.append(df2)

    df = df.reset_index(drop=True)

    for i in range(10010):
        if i > 0 and df.at[i, 'min'] > df.at[i-1, 'min']:
            df.at[i, 'min'] = df.at[i-1, 'min']

    
    print(df.to_string())

    # print best solution found:
    best = hof3.items[0]
    print("Best Ever Individual = ", best)
    print("Best Ever Fitness = ", best.fitness.values[0])

    print("--- TSP best route --- ")
    tsp.printResult(best)


    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'min':['mean', 'std']})
    print(df_promedios.to_string())

    x = df['gen'].unique()

    promedios = df_promedios['min']['mean'].values
    desviacion = df_promedios['min']['std'].values
    plt.plot(x, promedios, color='r')
    plt.plot(x, promedios - desviacion, linestyle = '--', color='b')
    plt.plot(x, promedios + desviacion, linestyle = '--', color='g')

    plt.show()

if __name__ == "__main__":
    main()
