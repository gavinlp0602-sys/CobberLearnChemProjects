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