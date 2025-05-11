import os
import subprocess

# Set the directory containing pcap files
pcap_dir = "/home/etbert/netDiffusion/NetDiffusion_Generator/data/unorder_pcap"  # Change this to your actual directory
output_dir = "/home/etbert/netDiffusion/NetDiffusion_Generator/data/fine_tune_pcaps"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each pcap file
for filename in os.listdir(pcap_dir):
    if filename.endswith(".pcap"):
        input_path = os.path.join(pcap_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Run reordercap
        subprocess.run(["reordercap", input_path, output_path], check=True)
        print(f"Reordered: {filename} -> {output_path}")

print("Processing complete.")
