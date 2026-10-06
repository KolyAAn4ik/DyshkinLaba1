import random
import numpy as np


class Individual:
    def __init__(self, genes):
        self.genes = np.array(genes)
        self.fitness = float('inf')


def variant_15_fitness(genes):
    """
    Целевая функция для Варианта 15: f(x) = sum(i * x_i^2), i=1..d
    Примечание: в Python индексация с 0, поэтому (i+1) * x[i]**2
    """
    return sum((i + 1) * (x ** 2) for i, x in enumerate(genes))


def initialize_population(pop_size, dim, bounds):
    """Инициализация в допустимой области"""
    lower, upper = bounds
    population = []
    for _ in range(pop_size):
        genes = np.random.uniform(lower, upper, dim)
        population.append(Individual(genes))
    return population


def tournament_selection(population, k=3):
    """Турнирная селекция"""
    selected = random.sample(population, k)
    return min(selected, key=lambda ind: ind.fitness)


def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """Скрещивание для вещественных векторов (BLX-alpha)"""
    genes1 = parent1.genes
    genes2 = parent2.genes
    offspring1_genes = []
    offspring2_genes = []

    for g1, g2 in zip(genes1, genes2):
        c_min = min(g1, g2)
        c_max = max(g1, g2)
        diff = c_max - c_min

        lower_bound = c_min - alpha * diff
        upper_bound = c_max + alpha * diff

        offspring1_genes.append(random.uniform(lower_bound, upper_bound))
        offspring2_genes.append(random.uniform(lower_bound, upper_bound))

    return Individual(np.array(offspring1_genes)), Individual(np.array(offspring2_genes))


def gaussian_mutation(individual, mutation_prob, sigma, bounds):
    """Мутация вещественных параметров (гауссова)"""
    lower, upper = bounds
    mutated_genes = individual.genes.copy()
    for i in range(len(mutated_genes)):
        if random.random() < mutation_prob:
            mutated_genes[i] += random.gauss(0, sigma)
            # Обработка выхода за границы (клиппинг)
            mutated_genes[i] = np.clip(mutated_genes[i], lower, upper)
    return Individual(mutated_genes)


def run_ga(pop_size, dim, bounds, max_generations, crossover_prob, mutation_prob, mutation_sigma, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = initialize_population(pop_size, dim, bounds)
    evaluations = 0

    # Оценка начальной популяции
    for ind in population:
        ind.fitness = variant_15_fitness(ind.genes)
        evaluations += 1

    best_individual = min(population, key=lambda ind: ind.fitness)

    for generation in range(max_generations):
        new_population = []

        # Элитизм: сохраняем лучшую особь
        new_population.append(best_individual)

        while len(new_population) < pop_size:
            # Селекция
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Рекомбинация
            if random.random() < crossover_prob:
                child1, child2 = blx_alpha_crossover(parent1, parent2)
            else:
                child1, child2 = Individual(parent1.genes.copy()), Individual(parent2.genes.copy())

            # Мутация
            child1 = gaussian_mutation(child1, mutation_prob, mutation_sigma, bounds)
            child2 = gaussian_mutation(child2, mutation_prob, mutation_sigma, bounds)

            # Оценка
            child1.fitness = variant_15_fitness(child1.genes)
            child2.fitness = variant_15_fitness(child2.genes)
            evaluations += 2

            new_population.extend([child1, child2])

        population = new_population[:pop_size]

        # Обновление лучшего решения
        current_best = min(population, key=lambda ind: ind.fitness)
        if current_best.fitness < best_individual.fitness:
            best_individual = Individual(current_best.genes.copy())
            best_individual.fitness = current_best.fitness

    return best_individual, evaluations