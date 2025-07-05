# Marketing-Budget-Optimization-GA
This repository implements a data-driven optimization pipeline to allocate marketing budgets across multiple channels using **Genetic Algorithms (GA)**. The goal is to **maximize return on investment (ROI)** using historical marketing performance data and evolutionary techniques.
# ====================== INSTALL & IMPORT DEPENDENCIES ======================
!pip install -q deap pandas matplotlib numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings("ignore")

# ====================== LOAD & PREPARE DATA ======================
from google.colab import files
uploaded = files.upload()

# After uploading, use the CSV
df = pd.read_csv(next(iter(uploaded)))
df.head()

# ====================== DATA CLEANING ======================
df = df.dropna()
channels = df['Channel'].unique()
channel_indices = {c: i for i, c in enumerate(channels)}
df['Channel_Index'] = df['Channel'].map(channel_indices)

# ====================== OBJECTIVE FUNCTION ======================
def fitness(individual):
    total_budget = 1_000_000  # or scale based on actual budget
    weights = np.array(individual) / sum(individual)
    roi = 0
    for idx, weight in enumerate(weights):
        roi += weight * df[df['Channel_Index'] == idx]['ROI'].mean()
    return roi,

# ====================== GENETIC ALGORITHM SETUP ======================
num_channels = len(channels)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_channels)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

# ====================== RUN GENETIC ALGORITHM ======================
population = toolbox.population(n=50)
NGEN = 50
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

top_ind = tools.selBest(population, k=1)[0]
weights = np.array(top_ind) / sum(top_ind)

# ====================== ALLOCATE BUDGET ======================
budget = 1_000_000
allocations = weights * budget

# Display optimal allocations
print("\n=== Optimal Budget Allocation ===")
for c, amt in zip(channels, allocations):
    print(f"{c}: ₹{amt:,.2f}")

# ====================== PLOT ALLOCATION ======================
plt.figure(figsize=(10, 6))
plt.bar(channels, allocations, color='skyblue')
plt.title("Optimized Marketing Budget Allocation")
plt.ylabel("Budget (INR)")
plt.xlabel("Channel")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

