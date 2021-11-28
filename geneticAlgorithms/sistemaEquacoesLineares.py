import random

POPULATION_SIZE = 200

GENES = '01'


def func1(x, y, z, w):
    return pow(x, 2) + pow(y, 3) + pow(z, 4) - pow(w, 5)


def func2(x, y, z, w):
    return pow(x, 2) + 3 * pow(z, 2) - w


def func3(x, y, z, w):
    return pow(z, 5) - y - 10


def func4(x, y, z, w):
    return pow(x, 4) - z + y * w


class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(cls):
        gnome_len = 64
        return [cls.mutated_genes() for _ in range(gnome_len)]

    def mate(self, par2):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob = random.random()
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutated_genes())

        return Individual(child_chromosome)

    def get_chromosome_value(self, position):
        if self.chromosome[16 * (position - 1)] == '1':
            return -1 * int(''.join(self.chromosome[16 * (position - 1) + 1:position * 16 - 8]), 2) + int(''.join(self.chromosome[position * 16 - 8:position * 16]), 2) / 256
        return int(''.join(self.chromosome[16 * (position - 1) + 1:position * 16 - 8]), 2) + int(''.join(self.chromosome[position * 16 - 8:position * 16]), 2) / 256

    def cal_fitness(self):
        sum = 0

        sum += abs(func1(self.get_chromosome_value(1), self.get_chromosome_value(2), self.get_chromosome_value(3),
                         self.get_chromosome_value(4)))
        sum += abs(func2(self.get_chromosome_value(1), self.get_chromosome_value(2), self.get_chromosome_value(3),
                         self.get_chromosome_value(4)))
        sum += abs(func3(self.get_chromosome_value(1), self.get_chromosome_value(2), self.get_chromosome_value(3),
                         self.get_chromosome_value(4)))
        sum += abs(func4(self.get_chromosome_value(1), self.get_chromosome_value(2), self.get_chromosome_value(3),
                         self.get_chromosome_value(4)))

        return sum


def main():
    global POPULATION_SIZE

    generation = 1

    found = False
    population = []

    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found and generation < 1000:

        population = sorted(population, key=lambda x: x.fitness)

        if population[0].fitness <= 0:
            break

        new_generation = []

        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print("Generation: {}\tString: {}\tFitness: {}". \
              format(generation,
                     "".join(population[0].chromosome),
                     population[0].fitness))

        generation += 1

    # print(population[0].chromosome)
    f1 = abs(func1(population[0].get_chromosome_value(1), population[0].get_chromosome_value(2),
                   population[0].get_chromosome_value(3), population[0].get_chromosome_value(4)))
    f2 = abs(func2(population[0].get_chromosome_value(1), population[0].get_chromosome_value(2),
                   population[0].get_chromosome_value(3), population[0].get_chromosome_value(4)))
    f3 = abs(func3(population[0].get_chromosome_value(1), population[0].get_chromosome_value(2),
                   population[0].get_chromosome_value(3), population[0].get_chromosome_value(4)))
    f4 = abs(func4(population[0].get_chromosome_value(1), population[0].get_chromosome_value(2),
                   population[0].get_chromosome_value(3), population[0].get_chromosome_value(4)))

    print("x: %.4f" % population[0].get_chromosome_value(1), "y: %.4f" % population[0].get_chromosome_value(2),
          "z: %.4f" % population[0].get_chromosome_value(3), "w: %.4f" % population[0].get_chromosome_value(4))
    print(f1, f2, f3, f4, f1 + f2 + f3 + f4)

    print("Generation: {}\tString: {}\tFitness: {}". \
          format(generation,
                 "".join(population[0].chromosome),
                 "{:.8f}".format(population[0].fitness)))


if __name__ == '__main__':
    main()
