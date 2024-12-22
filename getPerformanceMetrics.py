import json, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from satellite import Satellite
from constellation import Constellation
from scipy.stats import linregress
import tensorflow as tf

def simulateTest():
    test_size = 1000

    num_satellites = 100 # Initialize with 100 satellites
    satellites = [
        Satellite(
            longitude = np.random.uniform(0, 360),
            latitude = np.random.uniform(-90, 90),
            height = 0,
            speed = 0.5
        ) for _ in range(num_satellites)
    ]

    network = Constellation()
    data = []

    for _ in range(test_size):
        print("Starting test %d/%d" % (_, test_size))

        sat1 = int(np.random.uniform(0, num_satellites))
        sat2 = int(np.random.uniform(0, num_satellites))
        while sat1 == sat2:
            sat2 = int(np.random.uniform(0, num_satellites))

        results = network.compare_routing_methods(satellites, start_index=sat1, end_index=sat2)
        data.append(results)

    with open("output.json", "w") as save_file: 
        json.dump(data, save_file, indent=True)
    
    return data

def descriptiveStatistics(data: list) -> None:
    # Prepare data for DataFrame
    rows = []
    for entry in data:
        # Extract data for optimal and non-optimal
        optimal = entry['optimal']
        non_optimal = entry['non-optimal']

        # Add data for optimal
        rows.append({
            "Algorithm": "Multiagent Routing Algorithm",
            "Distance": optimal['distance'],
            "Num_Satellites": optimal['num_satellites'],
            "True_Distance": optimal['true_distance']
        })

        # Add data for non-optimal
        rows.append({
            "Algorithm": "Flooding Algorithm",
            "Distance": non_optimal['distance'],
            "Num_Satellites": non_optimal['num_satellites'],
            "True_Distance": non_optimal['true_distance']
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Compute descriptive statistics
    summary = df.groupby('Algorithm').agg(
        Distance_mean=('Distance', 'mean'),
        Distance_median=('Distance', 'median'),
        Distance_std=('Distance', 'std'),
        Num_Satellites_mean=('Num_Satellites', 'mean'),
        Num_Satellites_median=('Num_Satellites', 'median'),
        Num_Satellites_std=('Num_Satellites', 'std'),
        True_Distance=('True_Distance', 'mean')  # True distance is constant, so 'mean' suffices
    )

    # Calculate percentage differences
    distance_diff = (
        abs(summary.loc["Multiagent Routing Algorithm", "Distance_mean"] - summary.loc["Flooding Algorithm", "Distance_mean"])
        / summary.loc["Flooding Algorithm", "Distance_mean"]
    ) * 100

    satellites_diff = (
        abs(summary.loc["Multiagent Routing Algorithm", "Num_Satellites_mean"] - summary.loc["Flooding Algorithm", "Num_Satellites_mean"])
        / summary.loc["Flooding Algorithm", "Num_Satellites_mean"]
    ) * 100

    # Add percentage differences to the summary
    percentage_differences = pd.DataFrame(
        {
            "Metric": ["Distance Mean % Difference", "Num_Satellites Mean % Difference"],
            "Percentage Difference": [distance_diff, satellites_diff]
        }
    )

    # Display the table
    print("Descriptive Statistics:")
    print(summary.to_markdown())
    print("\nPercentage Differences:")
    print(percentage_differences.to_markdown())

def plotScatterPlot(data: list, output_filename="distance_comparison_scatterplot.png") -> None:
    # Prepare data for plotting
    optimal_true = []
    optimal_computed = []
    non_optimal_true = []
    non_optimal_computed = []

    for entry in data:
        optimal_entry = entry['optimal']
        non_optimal_entry = entry['non-optimal']

        # Extract true and computed distances for optimal
        optimal_true.append(optimal_entry['true_distance'])
        optimal_computed.append(optimal_entry['distance'])

        # Extract true and computed distances for non-optimal
        non_optimal_true.append(non_optimal_entry['true_distance'])
        non_optimal_computed.append(non_optimal_entry['distance'])

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    plt.scatter(optimal_true, optimal_computed, label='Multiagent Routing Algorithm', alpha=0.6, color='blue', s=20)
    plt.scatter(non_optimal_true, non_optimal_computed, label='Flooding Algorithm', alpha=0.6, color='purple', s=20)

    # Add a diagonal line for reference
    max_true_distance = max(max(optimal_true), max(non_optimal_true))
    plt.plot([0, max_true_distance], [0, max_true_distance], color='black', linestyle='--', linewidth=1, label='Reference Line')

    # Add labels, title, and legend
    plt.xlabel('True Distance (KM)', fontsize=12)
    plt.ylabel('Computed Distance (KM)', fontsize=12)
    plt.title('Distance vs True Distance Comparison Scatterplot', fontsize=14)
    plt.legend(title='Algorithm', fontsize=10)
    plt.grid(True)

    # Create the /performance directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "performance")
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plotDelayedPaths(data: list, output_flooding="flooding_delayed_paths.png", output_optimal="multiagent_delayed_paths.png") -> None:
    # Prepare data for plotting
    delay_types = ["low", "medium", "high"]
    colors = {"low": "green", "medium": "orange", "high": "red"}
    
    flooding_data = {"low": [], "medium": [], "high": []}
    optimal_data = {"low": [], "medium": [], "high": []}
    
    for entry in data:
        # Extract true distance
        true_distance_optimal = entry["optimal"]["true_distance"]
        true_distance_flooding = entry["non-optimal"]["true_distance"]
        
        # Extract delayed satellite counts
        for delay_type in delay_types:
            flooding_data[delay_type].append(
                (true_distance_flooding, entry["non-optimal"]["number_of_delayed_satellites"][delay_type])
            )
            optimal_data[delay_type].append(
                (true_distance_optimal, entry["optimal"]["number_of_delayed_satellites"][delay_type])
            )

    # Plot Flooding Algorithm
    fig, ax = plt.subplots(figsize=(10, 6))
    for delay_type in delay_types:
        distances, delays = zip(*flooding_data[delay_type])
        ax.scatter(distances, delays, label=f"{delay_type} delay", color=colors[delay_type], alpha=0.7)
        # Add correlation line
        if len(distances) > 1:
            slope, intercept, _, _, _ = linregress(distances, delays)
            ax.plot(distances, np.array(distances) * slope + intercept, color=colors[delay_type], linestyle="--", alpha=0.5)

    ax.set_xlabel("True Distance (KM)", fontsize=12)
    ax.set_ylabel("Number of Delayed Satellites", fontsize=12)
    ax.set_title("Flooding Algorithm: True Distance vs. Delayed Paths", fontsize=14)
    ax.legend()
    ax.grid(True)

    # Save Flooding Plot
    output_dir = os.path.join(os.getcwd(), "performance")
    os.makedirs(output_dir, exist_ok=True)
    flooding_path = os.path.join(output_dir, output_flooding)
    plt.tight_layout()
    plt.savefig(flooding_path)
    plt.close()
    print(f"Flooding plot saved to {flooding_path}")

    # Plot Optimal Algorithm
    fig, ax = plt.subplots(figsize=(10, 6))
    for delay_type in delay_types:
        distances, delays = zip(*optimal_data[delay_type])
        ax.scatter(distances, delays, label=f"{delay_type} delay", color=colors[delay_type], alpha=0.7)
        # Add correlation line
        if len(distances) > 1:
            slope, intercept, _, _, _ = linregress(distances, delays)
            ax.plot(distances, np.array(distances) * slope + intercept, color=colors[delay_type], linestyle="--", alpha=0.5)

    ax.set_xlabel("True Distance (KM)", fontsize=12)
    ax.set_ylabel("Number of Delayed Satellites", fontsize=12)
    ax.set_title("Multiagent Algorithm: True Distance vs. Delayed Paths", fontsize=14)
    ax.legend()
    ax.grid(True)

    # Save Optimal Plot
    optimal_path = os.path.join(output_dir, output_optimal)
    plt.tight_layout()
    plt.savefig(optimal_path)
    plt.close()
    print(f"Optimal plot saved to {optimal_path}")

def plotCongestedPaths(data: list, output_flooding="flooding_congested_paths.png", output_optimal="multiagent_congested_paths.png") -> None:
    # Prepare data for plotting
    congestion_types = ["low", "medium", "high"]
    colors = {"low": "green", "medium": "orange", "high": "red"}
    
    flooding_data = {"low": [], "medium": [], "high": []}
    optimal_data = {"low": [], "medium": [], "high": []}
    
    for entry in data:
        # Extract true distance
        true_distance_optimal = entry["optimal"]["true_distance"]
        true_distance_flooding = entry["non-optimal"]["true_distance"]
        
        # Extract congested satellite counts
        for congestion_type in congestion_types:
            flooding_data[congestion_type].append(
                (true_distance_flooding, entry["non-optimal"]["number_of_congested_satellites"][congestion_type])
            )
            optimal_data[congestion_type].append(
                (true_distance_optimal, entry["optimal"]["number_of_congested_satellites"][congestion_type])
            )

    # Plot Flooding Algorithm
    fig, ax = plt.subplots(figsize=(10, 6))
    for congestion_type in congestion_types:
        distances, congestions = zip(*flooding_data[congestion_type])
        ax.scatter(distances, congestions, label=f"{congestion_type} congestion", color=colors[congestion_type], alpha=0.7)
        # Add correlation line
        if len(distances) > 1:
            slope, intercept, _, _, _ = linregress(distances, congestions)
            ax.plot(distances, np.array(distances) * slope + intercept, color=colors[congestion_type], linestyle="--", alpha=0.5)

    ax.set_xlabel("True Distance (KM)", fontsize=12)
    ax.set_ylabel("Number of Congested Satellites", fontsize=12)
    ax.set_title("Flooding Algorithm: True Distance vs. Congested Paths", fontsize=14)
    ax.legend()
    ax.grid(True)

    # Save Flooding Plot
    output_dir = os.path.join(os.getcwd(), "performance")
    os.makedirs(output_dir, exist_ok=True)
    flooding_path = os.path.join(output_dir, output_flooding)
    plt.tight_layout()
    plt.savefig(flooding_path)
    plt.close()
    print(f"Flooding plot saved to {flooding_path}")

    # Plot Optimal Algorithm
    fig, ax = plt.subplots(figsize=(10, 6))
    for congestion_type in congestion_types:
        distances, congestions = zip(*optimal_data[congestion_type])
        ax.scatter(distances, congestions, label=f"{congestion_type} congestion", color=colors[congestion_type], alpha=0.7)
        # Add correlation line
        if len(distances) > 1:
            slope, intercept, _, _, _ = linregress(distances, congestions)
            ax.plot(distances, np.array(distances) * slope + intercept, color=colors[congestion_type], linestyle="--", alpha=0.5)

    ax.set_xlabel("True Distance (KM)", fontsize=12)
    ax.set_ylabel("Number of Congested Satellites", fontsize=12)
    ax.set_title("Optimal Algorithm: True Distance vs. Congested Paths", fontsize=14)
    ax.legend()
    ax.grid(True)

    # Save Optimal Plot
    optimal_path = os.path.join(output_dir, output_optimal)
    plt.tight_layout()
    plt.savefig(optimal_path)
    plt.close()
    print(f"Optimal plot saved to {optimal_path}")

if __name__ == "__main__":
    data = simulateTest()
    
    with open("output.json", "r") as load_file:
        data = json.load(load_file)

    descriptiveStatistics(data)
    plotScatterPlot(data)
    plotDelayedPaths(data)
    plotCongestedPaths(data)