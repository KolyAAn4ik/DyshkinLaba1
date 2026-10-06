import argparse
import csv
import time
import numpy as np
from genetic_algorithm import run_ga


def main():
    parser = argparse.ArgumentParser(description="Генетический алгоритм для Варианта 15")
    parser.add_argument("--pop_size", type=int, default=50, help="Размер популяции (мин. 30)")
    parser.add_argument("--dim", type=int, default=10, help="Размерность задачи (d)")
    parser.add_argument("--generations", type=int, default=100, help="Макс. число поколений")
    parser.add_argument("--crossover_prob", type=float, default=0.8, help="Вероятность скрещивания")
    parser.add_argument("--mutation_prob", type=float, default=0.1, help="Вероятность мутации гена")
    parser.add_argument("--mutation_sigma", type=float, default=0.5, help="Сигма (размах) мутации")
    parser.add_argument("--runs", type=int, default=20, help="Количество независимых запусков")
    parser.add_argument("--output", type=str, default="results.csv", help="Файл для сохранения результатов")

    args = parser.parse_args()

    bounds = (-10.0, 10.0)
    results = []
    best_fitnesses = []
    all_evals = []
    all_times = []

    print(f"Запуск {args.runs} независимых экспериментов...")

    for run in range(args.runs):
        seed = run  # Фиксация seed для воспроизводимости
        start_time = time.time()

        best_ind, evals = run_ga(
            pop_size=args.pop_size,
            dim=args.dim,
            bounds=bounds,
            max_generations=args.generations,
            crossover_prob=args.crossover_prob,
            mutation_prob=args.mutation_prob,
            mutation_sigma=args.mutation_sigma,
            seed=seed
        )

        exec_time = time.time() - start_time

        results.append({
            "run": run + 1,
            "seed": seed,
            "best_fitness": best_ind.fitness,
            "evaluations": evals,
            "time_sec": round(exec_time, 4)
        })

        best_fitnesses.append(best_ind.fitness)
        all_evals.append(evals)
        all_times.append(exec_time)

        print(f"Запуск {run + 1}/{args.runs}: Best Fitness = {best_ind.fitness:.6e}, Время = {exec_time:.2f}с")

    # Статистический анализ
    best_fitnesses = np.array(best_fitnesses)
    print("\n--- Анализ результатов ---")
    print(f"Лучший результат:   {np.min(best_fitnesses):.6e}")
    print(f"Худший результат:   {np.max(best_fitnesses):.6e}")
    print(f"Среднее:            {np.mean(best_fitnesses):.6e}")
    print(f"Медиана:            {np.median(best_fitnesses):.6e}")
    print(f"Ст. отклонение:     {np.std(best_fitnesses):.6e}")
    print(f"Среднее время:      {np.mean(all_times):.2f} сек")
    print(f"Среднее число выч.: {np.mean(all_evals):.0f}")

    # Сохранение в CSV
    with open(args.output, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["run", "seed", "best_fitness", "evaluations", "time_sec"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()