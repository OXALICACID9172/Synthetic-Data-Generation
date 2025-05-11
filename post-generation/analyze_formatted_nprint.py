import pandas as pd
import numpy as np
from collections import Counter

def analyze_nprint(formatted_nprint_path, sample_size=10):
    # Load the nPrint file (only the src_ip column to save memory)
    print("Loading nPrint file...")
    df = pd.read_csv(formatted_nprint_path, usecols=['src_ip'])
    print(f"Loaded {len(df)} packets.")

    # Step 1: Compute IP Distribution
    ip_counts = df['src_ip'].value_counts()
    ip_distribution = (ip_counts / ip_counts.sum()).to_dict()
    top_two_ips = dict(sorted(ip_distribution.items(), key=lambda x: x[1], reverse=True)[:2])
    print("\nTop 2 Source IPs and Their Percentages:")
    for ip, perc in top_two_ips.items():
        print(f"{ip}: {perc:.4f} ({ip_counts[ip]} packets)")

    # Step 2: Get Ordered IPs and Map to States
    ordered_ips = [ip for ip in df['src_ip'].drop_duplicates() if ip in top_two_ips]
    if len(ordered_ips) < 2:
        print("Error: Less than 2 unique IPs in top_two_ips.")
        return
    first_ip, second_ip = ordered_ips[0], ordered_ips[1]
    state_map = {first_ip: 0, second_ip: 1}
    print(f"\nState Mapping:\n0: {first_ip}\n1: {second_ip}")

    # Step 3: Compute Transition Counts
    transition_counts = {
        (first_ip, first_ip): 0,
        (first_ip, second_ip): 0,
        (second_ip, first_ip): 0,
        (second_ip, second_ip): 0
    }
    src_ips = df['src_ip'].values
    for i in range(1, len(src_ips)):
        prev_ip, curr_ip = src_ips[i-1], src_ips[i]
        if prev_ip in top_two_ips and curr_ip in top_two_ips:
            transition_counts[(prev_ip, curr_ip)] += 1

    print("\nTransition Counts:")
    for (from_ip, to_ip), count in transition_counts.items():
        print(f"{from_ip} -> {to_ip}: {count}")

    # Step 4: Build Transition Matrix
    total_from_first = transition_counts[(first_ip, first_ip)] + transition_counts[(first_ip, second_ip)]
    total_from_second = transition_counts[(second_ip, first_ip)] + transition_counts[(second_ip, second_ip)]
    
    transition_matrix = np.array([
        [transition_counts[(first_ip, first_ip)] / total_from_first if total_from_first > 0 else 0,
         transition_counts[(first_ip, second_ip)] / total_from_first if total_from_first > 0 else 0],
        [transition_counts[(second_ip, first_ip)] / total_from_second if total_from_second > 0 else 0,
         transition_counts[(second_ip, second_ip)] / total_from_second if total_from_second > 0 else 0]
    ])
    print("\nTransition Matrix:")
    print(f"From {first_ip} (0): [To 0: {transition_matrix[0,0]:.4f}, To 1: {transition_matrix[0,1]:.4f}]")
    print(f"From {second_ip} (1): [To 0: {transition_matrix[1,0]:.4f}, To 1: {transition_matrix[1,1]:.4f}]")

    # Step 5: Generate Sample Sequence
    current_state = 0  # Start with first_ip
    synthetic_sequence = [current_state]
    for _ in range(sample_size - 1):
        current_state = np.random.choice([0, 1], p=transition_matrix[current_state])
        synthetic_sequence.append(current_state)
    print(f"\nSample Synthetic Sequence ({sample_size} steps):")
    print(synthetic_sequence)

if __name__ == "__main__":
    # Replace with your actual file path
    formatted_nprint_path = "/home/etbert/netDiffusion/NetDiffusion_Generator/post-generation/correct_format.nprint"
    analyze_nprint(formatted_nprint_path, sample_size=10)
