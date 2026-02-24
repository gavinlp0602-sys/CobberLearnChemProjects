import random
import string
import os
import matplotlib.pyplot as plt
from datetime import datetime


# --- 1. CORE LOGIC FUNCTIONS ---

def calculate_match_score(input_string, target_phrase):
    """Compares input to target; 1 point per correct char, max 28."""
    points = 0
    comparison_length = min(len(input_string), len(target_phrase), 28)
    for i in range(comparison_length):
        if input_string[i] == target_phrase[i]:
            points += 1
    return points


def mutate_string(input_string, chance=0.05):
    """5% chance to change any character into a random letter or space."""
    possible_chars = string.ascii_letters + " " + "-" +"."
    mutated_list = list(input_string)
    for i in range(len(mutated_list)):
        if random.random() < chance:
            mutated_list[i] = random.choice(possible_chars)
    return "".join(mutated_list)


def create_fitness_plot(history, target_display, folder="evolution_results"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(folder, f"fitness_{timestamp}.png")

    gens, scores = zip(*history)
    plt.figure(figsize=(12, 6))
    plt.plot(gens, scores, color='#9467bd', linewidth=2, marker='.', markersize=4, alpha=0.7)

    plt.title(f"Evolution of: '{target_display.strip()}'\nFinal Generation: {gens[-1]}")
    plt.xlabel('Generation')
    plt.ylabel('Match Score (Max 28)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0, 30)

    plt.savefig(full_path)
    plt.close()
    print(f"\n[SYSTEM] Plot successfully saved to: {full_path}")


# --- 2. USER INPUT

print("--- EVOLUTION SIMULATOR ---")
user_target = input("Enter a target phrase (max 28 chars): ")

# Clean up input: truncate if over 28, pad with spaces if under
target = user_target.ljust(28)[:28]
current_best = "".join(random.choice(string.ascii_letters + " ") for _ in range(28))
generation = 0
history = []

print(f"\nEvolving toward: '{target}'")
print("-" * 30)

# --- 3. THE SIMULATION LOOP ---

# --- 3. THE SIMULATION LOOP ---

while calculate_match_score(current_best, target) < 28:
    generation += 1

    # Generate 100 variations
    population = [mutate_string(current_best) for _ in range(100)]

    # Survival of the fittest
    current_best = max(population, key=lambda s: calculate_match_score(s, target))
    current_score = calculate_match_score(current_best, target)

    history.append((generation, current_score))

    # Print every 5 generations to keep the console clean
    if generation % 5 == 0 or current_score == 28:
        print(f"Gen {generation:3} | Score: {current_score:2}/28 | '{current_best}'")

# --- 3. THE REPORTING PHASE ---

# Now that the loop is finished, we save the file
create_fitness_plot(history, target)
print("="*40)
print("Simulation complete.")