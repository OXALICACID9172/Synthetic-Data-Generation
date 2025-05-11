import os
import subprocess

input_dir = "/home/rishabh/MTP-1/Labels_Pcap"
output_dir = "/home/etbert/netDiffusion/NetDiffusion_Generator/data/raw_data"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".pcap"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        cmd = ["tcprewrite", "--enet-vlan=del", "--infile", input_path, "--outfile", output_path]
        subprocess.run(cmd)
