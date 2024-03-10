import math
import itertools
import time
import random

def calculate_distance(location1, location2):
    """
    Calculate the Euclidean distance between two locations.

    Args:
        location1: Tuple representing (x1, y1) coordinates of the first location.
        location2: Tuple representing (x2, y2) coordinates of the second location.

    Returns:
        Euclidean distance between the two locations in meters.
    """
    x1, y1 = location1
    x2, y2 = location2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def total_distance(path, locations):
    """Calculate total distance of a path visiting all locations."""
    total = 0
    for i in range(len(path) - 1):
        total += calculate_distance(locations[path[i]], locations[path[i + 1]])
    # Add distance from last location back to the start
    total += calculate_distance(locations[path[-1]], locations[path[0]])
    return total

def generate_feeding_locations(n):
    """
    Generate 'n' random feeding locations within a 100,000 square meter area.

    Args:
        n: Number of feeding locations to generate.

    Returns:
        List of tuples representing (x, y) coordinates of feeding locations.
    """
    max_x = 316  # Square root of 100,000 for x-coordinate limit
    max_y = 316  # Square root of 100,000 for y-coordinate limit

    feeding_locations = []
    for _ in range(n):
        x = random.randint(0, max_x-1)
        y = random.randint(0, max_y-1)
        feeding_locations.append((x, y))

    return feeding_locations

def brute_force_shortest_paths(locations, initial_path, num_paths=5):
    """
    Brute force method to find the top shortest paths to visit all locations.

    Args:
        locations: List of (x, y) coordinates of feeding locations.
        initial_path: Initial random path to start with.
        num_paths: Number of top shortest paths to find.

    Returns:
        List of tuples, each containing a shortest path, its total distance, and time taken.
    """
    num_locations = len(locations)
    if num_locations <= 1:
        return [(locations, 0, 0)]

    # Create list of indices representing the locations
    indices = list(range(num_locations))

    initial_distance = total_distance(initial_path, locations)

    paths = []
    shortest_paths = []

    start_time = time.time()

    # Generate all permutations of the locations
    for perm in itertools.permutations(indices):
        current_distance = total_distance(perm, locations)
        paths.append((list(perm), current_distance))

    # Sort paths by distance
    paths.sort(key=lambda x: x[1])

    for i in range(min(num_paths, len(paths))):
        shortest_paths.append((paths[i][0][:], paths[i][1]))  # Copy path before appending

    end_time = time.time()

    # Convert indices back to locations for shortest paths
    shortest_paths_locations = [[locations[i] for i in path] for path, distance in shortest_paths]

    # Calculate time taken
    time_taken = end_time - start_time

    return shortest_paths_locations, shortest_paths, time_taken, initial_distance

def greedy_shortest_paths(locations, initial_path, num_paths=5):
    """
    Greedy method to find the top shortest paths to visit all locations.

    Args:
        locations: List of (x, y) coordinates of feeding locations.
        initial_path: Initial random path to start with.
        num_paths: Number of top shortest paths to find.

    Returns:
        List of tuples, each containing a shortest path, its total distance, and time taken.
    """
    num_locations = len(locations)
    if num_locations <= 1:
        return [(locations, 0, 0)]

    # Create list of indices representing the locations
    indices = list(range(num_locations))

    initial_distance = total_distance(initial_path, locations)

    paths = []
    shortest_paths = []

    start_time = time.time()

    # Greedy algorithm
    for _ in range(min(num_paths, num_locations)):
        current_path = []
        remaining_locations = indices.copy()
        current_location = random.choice(remaining_locations)
        remaining_locations.remove(current_location)
        current_path.append(current_location)

        while remaining_locations:
            next_location = min(remaining_locations, key=lambda x: calculate_distance(locations[current_location], locations[x]))
            current_path.append(next_location)
            remaining_locations.remove(next_location)
            current_location = next_location

        current_distance = total_distance(current_path, locations)
        paths.append((current_path, current_distance))

    # Sort paths by distance
    paths.sort(key=lambda x: x[1])

    for i in range(min(num_paths, len(paths))):
        shortest_paths.append((paths[i][0][:], paths[i][1]))  # Copy path before appending

    end_time = time.time()

    # Convert indices back to locations for shortest paths
    shortest_paths_locations = [[locations[i] for i in path] for path, distance in shortest_paths]

    # Calculate time taken
    time_taken = end_time - start_time

    return shortest_paths_locations, shortest_paths, time_taken, initial_distance

def dynamic_programming_shortest_paths(locations):
    """
    Dynamic Programming method to find the shortest path to visit all locations.

    Args:
        locations: List of (x, y) coordinates of feeding locations.

    Returns:
        Tuple containing the shortest path, its total distance.
    """
    num_locations = len(locations)
    if num_locations <= 1:
        return locations, 0

    # Create list of indices representing the locations
    indices = list(range(num_locations))

    # Dictionary to store calculated distances
    memo = {}

    def dp(current_location, remaining_locations):
        # Base case: If no remaining locations, return 0 (distance from current_location to itself)
        if not remaining_locations:
            return 0

        # Create a tuple for memoization
        state = (current_location, tuple(remaining_locations))

        # If already calculated, return the memoized result
        if state in memo:
            return memo[state]

        # Initialize minimum distance to infinity
        min_distance = float('inf')

        # Try all remaining locations as the next stop
        for next_location in remaining_locations:
            new_remaining = remaining_locations.copy()
            new_remaining.remove(next_location)

            # Calculate distance for the current path
            distance = calculate_distance(locations[current_location], locations[next_location])

            # Recur for the remaining path
            remaining_distance = dp(next_location, new_remaining)

            # Update total distance
            total = distance + remaining_distance

            # Update minimum distance
            min_distance = min(min_distance, total)

        # Memoize the result
        memo[state] = min_distance
        return min_distance

    # Initialize variables for shortest path
    shortest_path = None
    min_total_distance = float('inf')

    # Call the dynamic programming function for all possible starting locations
    for start_location in indices:
        remaining = indices.copy()
        remaining.remove(start_location)
        distance = dp(start_location, remaining)

        # Check if this path is shorter than the minimum found so far
        if distance < min_total_distance:
            min_total_distance = distance
            shortest_path = [start_location] + remaining

    return shortest_path, min_total_distance

def top_k_shortest_paths(locations, k=5):
    """
    Find the top k shortest paths to visit all locations using dynamic programming.

    Args:
        locations: List of (x, y) coordinates of feeding locations.
        k: Number of top shortest paths to find.

    Returns:
        List of tuples, each containing a shortest path and its total distance.
    """
    all_permutations = list(itertools.permutations(range(len(locations))))
    all_paths = [(list(perm), total_distance(perm, locations)) for perm in all_permutations]
    all_paths.sort(key=lambda x: x[1])
    
    top_k_paths = all_paths[:k]
    return top_k_paths

def monte_carlo(locations, num_iterations=1000):
    """Monte Carlo method to find shortest path."""
    start_time = time.time()
    n = len(locations)
    shortest_path = None
    shortest_distance = float('inf')
    for _ in range(num_iterations):
        path = [0] + random.sample(range(1, n), n - 1)
        dist = total_distance(path, locations)
        if dist < shortest_distance:
            shortest_distance = dist
            shortest_path = path
    end_time = time.time()
    return shortest_path, shortest_distance, end_time - start_time

n = 6  # Number of feeding locations
locations = generate_feeding_locations(n)
initial_path = list(range(n))
random.shuffle(initial_path)
print(locations)

print("---- Brute Force Search ----")
top_shortest_paths_bf, top_shortest_paths_info_bf, time_taken_bf, initial_distance_bf = brute_force_shortest_paths(locations, initial_path, num_paths=5)
print("Initial Random Path Distance (Brute Force):", initial_distance_bf)
print("Initial Random Path (Brute Force):", [locations[i] for i in initial_path])
print("Top 5 Shortest Paths to Visit All Locations (Brute Force):")
for i, (path, distance) in enumerate(top_shortest_paths_info_bf, 1):
    print(f"Path {i} - Distance: {distance}")
    for j, loc in enumerate(path, 1):
        print(f"Step {j}: {locations[loc]}")
    print()
print("Time Taken (Brute Force):", time_taken_bf, "seconds")
print()

# Greedy Search
print("---- Greedy Search ----")
top_shortest_paths_greedy, top_shortest_paths_info_greedy, time_taken_greedy, initial_distance_greedy = greedy_shortest_paths(locations, initial_path, num_paths=5)
print("Initial Random Path Distance (Greedy):", initial_distance_greedy)
print("Initial Random Path (Greedy):", [locations[i] for i in initial_path])
print("Top 5 Shortest Paths to Visit All Locations (Greedy):")
for i, (path, distance) in enumerate(top_shortest_paths_info_greedy, 1):
    print(f"Path {i} - Distance: {distance}")
    for j, loc in enumerate(path, 1):
        print(f"Step {j}: {locations[loc]}")
    print()
print("Time Taken (Greedy):", time_taken_greedy, "seconds")
print()

print("---- Dynamic Programming ----")
start_time_top5 = time.time()
    top_shortest_paths_dp = top_k_shortest_paths(locations, k=5)
    end_time_top5 = time.time()
    time_taken_top5 = end_time_top5 - start_time_top5

    print("\nTop 5 Shortest Paths (Dynamic Programming):")
    for i, (path, distance) in enumerate(top_shortest_paths_dp, 1):
        print(f"Path {i} - Distance: {distance}")
        for j, loc in enumerate(path, 1):
            print(f"Step {j}: {locations[loc]}")
        print()
    print("Time Taken for Top 5 (Dynamic Programming):", time_taken_top5, "seconds")

print("\nMonte Carlo Method:")
path, distance, time_taken = monte_carlo(locations, num_iterations=10000)
print(f"Shortest Path: {path}")
print(f"Shortest Distance: {distance:.2f} meters")
print(f"Time taken: {time_taken:.6f} seconds")
