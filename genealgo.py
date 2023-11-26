from voxels import Voxel, VoxelMatrix
from beams import Beam, BeamSet
import math
import numpy as np
import random
import matplotlib.pyplot as plt

def calculate_beam_path(target, angle, width, height):
    path = set()

    angle_rad = math.radians(angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    x, y = target
    while 0 <= x < width and 0 <= y < height:
        path.add((int(x), int(y)))
        x += dx
        y += dy

    x, y = target
    while 0 <= x < width and 0 <= y < height:
        path.add((int(x), int(y)))
        x -= dx
        y -= dy

    return path


def aim_beams_at_voxelmatrix(beam_set, voxel_matrix, intensity):
    intensity_matrix = [[0 for _ in range(voxel_matrix.width)] for _ in range(voxel_matrix.height)]

    for beam in beam_set.beams:
        beam_path = calculate_beam_path(beam.target, beam.angle, voxel_matrix.width, voxel_matrix.height)
        for x, y in beam_path:
            intensity_matrix[y][x] += intensity 

    return intensity_matrix

def calculate_score(voxel_matrix, beam_set, intensity, weights):
    intensity_matrix = aim_beams_at_voxelmatrix(beam_set, voxel_matrix, intensity)
    target_matrix = voxel_matrix.matrix

    score = 0
    for i in range(voxel_matrix.height):
        for j in range(voxel_matrix.width):
            voxel = target_matrix[i][j]
            target = voxel.target
            type = voxel.voxel_type
            dose = intensity_matrix[i][j]

            if (type == 'PV'):
                score += weights['PV'] * abs(dose - target)
            elif (type == 'OA'):
                if dose > target:
                    score += weights['OA'] * (dose - target)
            elif (type == 'NT'):
                if dose > target:
                    score += weights['NT'] * (dose - target)

    return score

def initialise(voxel_matrix, n_init_beams=10, n_init_sets=50):
    population = []

    for _ in range(n_init_sets):
        beam_set = BeamSet()
        for _ in range(n_init_beams):
            angle = random.uniform(0, 360)
            target_x = random.randint(0, voxel_matrix.height - 1)
            target_y = random.randint(0, voxel_matrix.width - 1)
            beam_set.add_beam((target_x, target_y), angle)
        population.append(beam_set)
    return population

def exponential_ranking_selection(beam_sets, scores, c=0.99):
    if not 0 < c < 1:
        raise ValueError("c must be between 0 and 1")
    ranked_indices = np.argsort(scores).tolist()
    N = len(beam_sets)
    probabilities = [(1 - c) / (1 - c**N) * c**i for i in range(N)]
    probabilities = [probabilities[ranked_indices.index(i)] for i in range(N)]
    sampled_indices = np.random.choice(N, size=N, p=probabilities)
    sampled_beam_sets = [beam_sets[i] for i in sampled_indices]

    return sampled_beam_sets

def evaluate_scores(voxel_matrix, population, intensity, weights):
    scores = []
    for beam_set in population:
        score = calculate_score(voxel_matrix, beam_set, intensity, weights)
        scores.append(score)
        
    return scores

def select(population, scores):
    sampled = exponential_ranking_selection(population, scores)

    return sampled
 
def mutation(population, noise_std_angle, noise_std_target, prob, voxel_matrix):
    for beam_set in population:
        for beam in beam_set.beams:
            if random.random() < prob:
                beam.mutate_angle(noise_std_angle)
            if random.random() < prob:
                beam.mutate_target(noise_std_target, voxel_matrix)

    return population

def crossover(population):
    new_population = []
    for i in range(0, len(population), 2):
        if i + 1 < len(population):
            beam_set_1 = population[i].beams
            beam_set_2 = population[i + 1].beams
            ratio = len(beam_set_2) / len(beam_set_1)
            crossover_point_1 = random.randint(1, len(beam_set_1) - 1)
            crossover_point_2 = int(crossover_point_1 * ratio)

            # Creating offspring by swapping beams at the crossover point
            new_beam_set_1 = beam_set_1[:crossover_point_1] + beam_set_2[crossover_point_2:]
            new_beam_set_2 = beam_set_2[:crossover_point_2] + beam_set_1[crossover_point_1:]
            new_population.append(BeamSet(new_beam_set_1))
            new_population.append(BeamSet(new_beam_set_2))
        else:
            new_population.append(population[i].beams)

    return new_population    

def plot_avg_scores(avg_scores):
    # Assuming avg_scores is a list of average scores
    plt.plot(avg_scores)
    plt.xlabel('Generation')
    plt.ylabel('Average score')
    plt.title('Average scores across generations')
    plt.grid(True)
    plt.show()

def gene_algo(voxel_matrix, n_generations, intensity, weights, noise_std_angle = 5, noise_std_target = 2, mutate_prob = 0.05):
    population = initialise(voxel_matrix)
    scores_history = []
    for i in range(n_generations):
        scores = evaluate_scores(voxel_matrix, population, intensity, weights)
        avg_score = np.average(scores)
        print(f'Gen #%d/%d' % (i, n_generations), avg_score)
        scores_history.append(avg_score)
        population = select(population, scores)
        population = crossover(population)
        population = mutation(population, noise_std_angle, noise_std_target, mutate_prob, voxel_matrix)

    plot_avg_scores(scores_history)
    return population

voxel_matrix = VoxelMatrix()
voxel_matrix.read_from_file('voxel_data_2.txt')

population = gene_algo(voxel_matrix, 50, 1, {'OA': 1, 'PV': 1, 'NT': 1}, 5, 4, 0.1)