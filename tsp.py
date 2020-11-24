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


class KnapsackProblem:

    def __init__(self):
        self.items = []
        self.maxCapacikty = 0
        self.initData()

    def __len__(self):
        return len(self.items)

    def initData(self):

        self.items = [
            ("1", 23, 92),
            ("2", 31, 57),
            ("3", 29, 49),
            ("4", 44, 68),
            ("5", 53, 60),
            ("6", 38, 43),
            ("7", 63, 67),
            ("8", 85, 84),
            ("9", 89, 87),
            ("10", 82, 72)     
        ]

        self.maxWeight = 165

    def getValue(self, zeroOneList):

        totalWeight = totalValue = 0

        for i in range(len(zeroOneList)):
            item, weight, value = self.items[i]
            totalWeight += zeroOneList[i] * weight
            totalValue += zeroOneList[i] * value
        
        if totalWeight<= 165 :
            return totalValue
        else:
            return max(0, totalValue + 3*(165-totalWeight))


    def printItems(self, binaryList):

        totalWeight = totalValue = 0

        for i in range(len(binaryList)):
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxWeight:
                if binaryList[i] > 0:
                    totalWeight += weight
                    totalValue += value
                    print("Item {}: weight = {}, value = {}, current knapsack weight = {}, current knapsack value = {}".format(item, weight, value, totalWeight, totalValue))
        
        print("Total weight = ", totalWeight)
        print("Total value = ", totalValue)

#create instance of knapsack problem class
knapsack = KnapsackProblem()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


# evaluation function
def func_eval(individual):
    return knapsack.getValue(individual),  


toolbox.register("select", tools.selRoulette)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("evaluate", func_eval)

toolbox.register("attribute", random.randint, a=0, b=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
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

    print("--- Knapsack Items Chosen --- ")
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

    print("--- Knapsack Items Chosen --- ")
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

    print("--- Knapsack Items Chosen --- ")
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


    