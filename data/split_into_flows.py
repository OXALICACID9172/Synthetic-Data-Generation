import os
from scapy.all import rdpcap, wrpcap, TCP, UDP, IP
from collections import defaultdict

def extract_flows_from_pcap(pcap_path, output_dir, min_packets=10, protocol="both"):
    """Extracts flows from a single pcap file based on protocol and saves them if they meet the min_packets threshold."""
    packets = rdpcap(pcap_path)
    flows = defaultdict(list)
    
    for pkt in packets:
        if IP in pkt:
            if protocol == "tcp" and not pkt.haslayer(TCP):
                continue
            elif protocol == "udp" and not pkt.haslayer(UDP):
                continue
            elif protocol not in ["tcp", "udp", "both"]:
                raise ValueError("Invalid protocol filter. Use 'tcp', 'udp', or 'both'.")

            src, dst = pkt[IP].src, pkt[IP].dst
            if TCP in pkt or UDP in pkt:
                sport = pkt.sport
                dport = pkt.dport
                flow_key = (src, dst, sport, dport)
                flows[flow_key].append(pkt)

    # Get base name without extension
    pcap_name = os.path.splitext(os.path.basename(pcap_path))[0]
    flow_output_dir = os.path.join(output_dir, pcap_name)
    os.makedirs(flow_output_dir, exist_ok=True)

    # Save each flow that meets the minimum packet requirement
    for i, (key, packets) in enumerate(flows.items(), start=1):
        if len(packets) >= min_packets:
            flow_file = os.path.join(flow_output_dir, f"{pcap_name}_{i:05}.pcap")
            wrpcap(flow_file, packets)
            print(f"Saved {flow_file} with {len(packets)} packets.")

def process_pcap_directory(input_dir, output_dir, min_packets=10, protocol="both"):
    """Processes all pcap files in a directory with protocol filtering."""
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.endswith(".pcap"):
            pcap_path = os.path.join(input_dir, file)
            print(f"Processing {pcap_path} with protocol={protocol}...")
            extract_flows_from_pcap(pcap_path, output_dir, min_packets, protocol)

# Example usage
input_directory = "/home/etbert/netDiffusion/NetDiffusion_Generator/data/raw_data"
output_directory = "/home/etbert/netDiffusion/NetDiffusion_Generator/data/unorder_pcap"
min_packets_per_flow = 1
protocol_filter = "both"  # Options: "tcp", "udp", "both"

process_pcap_directory(input_directory, output_directory, min_packets_per_flow, protocol_filter)
