import random
import string
import matplotlib.pyplot as plt
import os

# 1. The Scoring Function
def calculate_match_score(input_string, target_phrase):
    points = 0
    comparison_length = min(len(input_string), len(target_phrase), 28)
    for i in range(comparison_length):
        if input_string[i] == target_phrase[i]:
            points += 1
    return points

# 2. The Mutation Function
def mutate_string(input_string, chance=0.05):
    possible_chars = string.ascii_letters + "-" + "." + " "
    mutated_list = list(input_string)
    for i in range(len(mutated_list)):
        if random.random() < chance:
            mutated_list[i] = random.choice(possible_chars)
    return "".join(mutated_list)

# --- The Simulation Loop ---
target = "I can count to twenty-eight." # Length 28
current_best = " " * len(target)      # Start with a blank string
generation = 0

print(f"Target: '{target}'\n" + "-"*30)

while calculate_match_score(current_best, target) < len(target):
    generation += 1
    candidates = []

    # Create 100 mutated versions of the current best
    for _ in range(100):
        candidates.append(mutate_string(current_best))

    # Pick the one with the highest score
    # This uses a 'key' to sort the list by our scoring function
    current_best = max(candidates, key=lambda s: calculate_match_score(s, target))

    # Show progress every 10 generations (or when we win)
    if generation % 10 == 0 or calculate_match_score(current_best, target) == len(target):
        score = calculate_match_score(current_best, target)
        print(f"Gen {generation:3} | Score: {score}/{len(target)} | Best: '{current_best}'")


def create_fitness_plot(history, folder_name="evolution_results", file_name="fitness_graph.png"):
    """
    Creates the directory if it doesn't exist and saves the plot.
    """
    # 1. Create the directory if it's not already there
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")

    # 2. Define the full path
    full_path = os.path.join(folder_name, file_name)

    # 3. Generate the plot
    generations, scores = zip(*history)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, scores, color='#2ca02c', marker='o', markersize=2)

    plt.title('Evolutionary Algorithm: Fitness Over Time')
    plt.xlabel('Generation')
    plt.ylabel('Score (out of 28)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. Save to the specific path
    plt.savefig(full_path)
    plt.close()

    print(f"Plot successfully saved to: {full_path}")