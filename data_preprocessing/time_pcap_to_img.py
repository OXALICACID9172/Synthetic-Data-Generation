import os
import subprocess
from PIL import Image
import pandas as pd
import numpy as np
from scapy.all import rdpcap

def ip_to_binary(ip_address):
    # Split the IP address into four octets
    octets = ip_address.split(".")
    # Convert each octet to binary form and pad with zeros
    binary_octets = []
    for octet in octets:
        binary_octet = bin(int(octet))[2:].zfill(8)
        binary_octets.append(binary_octet)

    # Concatenate the four octets to get the 32-bit binary representation
    binary_ip_address = "".join(binary_octets)
    return binary_ip_address
def binary_to_ip(binary_ip_address):
    # Check that the input binary string is 32 bits
    if len(binary_ip_address) != 32:
        raise ValueError("Input binary string must be 32 bits")
    # Split the binary string into four octets
    octets = [binary_ip_address[i:i+8] for i in range(0, 32, 8)]
    # Convert each octet from binary to decimal form
    decimal_octets = [str(int(octet, 2)) for octet in octets]

    # Concatenate the four decimal octets to get the IP address string
    ip_address = ".".join(decimal_octets)
    return ip_address

# Convert the DataFrame into a PNG image
def dataframe_to_png(df, output_file):
    width, height = df.shape[1], df.shape[0]
    padded_height = 1024
    print(output_file)
    # Convert DataFrame to numpy array and pad with blue pixels
    np_img = np.full((padded_height, width, 4), (0, 0, 255, 255), dtype=np.uint8)
    np_df = np.array(df.applymap(np.array).to_numpy().tolist())
    np_img[:height, :, :] = np_df
    print(np_img.shape)
    #print(np_img)
    # Create a new image with padded height filled with blue pixels
    img = Image.fromarray(np_img, 'RGBA')
    #print(img)
    # Check if file exists and generate a new file name if necessary
    file_exists = True
    counter = 1
    file_path, file_extension = os.path.splitext(output_file)
    while file_exists:
        if os.path.isfile(output_file):
            output_file = f"{file_path}_{counter}{file_extension}"
            counter += 1
        else:
            file_exists = False
    print(output_file)
    img.save(output_file)


def rgba_to_ip(rgba):
    ip_parts = tuple(map(str, rgba))
    ip = '.'.join(ip_parts)
    return ip
    
def int_to_rgba(A):
    if A == 1:
        rgba = (255, 0, 0, 255)
    elif A == 0:
        rgba = (0, 255, 0, 255)
    elif A == -1:
        rgba = (0, 0, 255, 255)
    elif A > 1:
        rgba = (255, 0, 0, A)
    elif A < -1:
        rgba = (0, 0, 255, abs(A))
    else:
        rgba = None
    return rgba

def rgba_to_int(rgba):
    if rgba == (255, 0, 0, 255):
        A = 1
    elif rgba == (0, 255, 0, 255):
        A = 0
    elif rgba == (0, 0, 255, 255):
        A = -1
    elif rgba[0] == 255 and rgba[1] == 0 and rgba[2] == 0:
        A = rgba[3]
    elif rgba[0] == 0 and rgba[1] == 0 and rgba[2] == 255:
        A = -rgba[3]
    else:
        A = None
    return A
# define function to split binary string into individual bits
def split_bits(s):
    return [int(b) for b in s]

data_dir = '../data/fine_tune_pcaps'
for i in os.listdir(data_dir):
    if 'pcap' in i:
        pcap_path = data_dir + '/' + i
        nprint_path = '../data/preprocessed_fine_tune_nprints/' + i.split('.pcap')[0] + '.nprint'
        
        print(f"Processing: {pcap_path}")
        
        # Run original nPrint command
        print('Creating nPrint for pcap')
        subprocess.run(f'nprint -F -1 -P {pcap_path} -4 -i -6 -t -u -p 0 -c 1024 -W {nprint_path}', shell=True)
        
        # Extract timestamps from PCAP using scapy
        print('Extracting and binarizing relative timestamps from PCAP')
        try:
            # Read the PCAP file
            packets = rdpcap(pcap_path)
            
            # Calculate relative timestamps (time since start of capture)
            if packets:
                start_time = float(packets[0].time)
                rel_timestamps = [float(packet.time) - start_time for packet in packets]
            else:
                rel_timestamps = []
            
            # Read the nPrint file
            nprint_df = pd.read_csv(nprint_path)
            
            # Add relative timestamp columns in binary format if packet counts match
            if len(rel_timestamps) > 1024:
                rel_timestamps = rel_timestamps[:1024]
            print("len_nprint:", len(nprint_df))
            print('len_rel_time:', len(rel_timestamps))
            if len(nprint_df) == len(rel_timestamps):
                # Convert each timestamp to binary representation
                # Number of bits to represent timestamp (adjust as needed)
                timestamp_bits = 32
                
                for idx, timestamp in enumerate(rel_timestamps):
                    # Convert timestamp to integer microseconds for precision
                    ts_microsec = int(timestamp * 1000000)
                    
                    # Convert to binary (removing '0b' prefix)
                    bin_timestamp = bin(ts_microsec)[2:]
                    
                    # Pad with leading zeros to ensure consistent length
                    bin_timestamp = bin_timestamp.zfill(timestamp_bits)
                    
                    # Add each bit as a separate column
                    for bit_idx, bit in enumerate(bin_timestamp):
                        col_name = f'rel_timestamp_{bit_idx}'
                        
                        # Create column if it doesn't exist
                        if col_name not in nprint_df.columns:
                            nprint_df[col_name] = -1  # Initialize with -1 (nPrint convention for missing)
                        
                        # Set the value for this packet
                        nprint_df.at[idx, col_name] = int(bit)
                
                # Save the updated nPrint file
                nprint_df.to_csv(nprint_path, index=False)
                print(f"Added binary-encoded relative timestamp columns to {nprint_path}")
            else:
                print(f"Warning: Number of packets in nPrint file ({len(nprint_df)}) doesn't match timestamps ({len(rel_timestamps)})")
        except Exception as e:
            print(f"Error processing timestamps for {pcap_path}: {e}")
            

nprint_dir = '../data/preprocessed_fine_tune_nprints'
for i in os.listdir(nprint_dir):
    if 'nprint' in i:
        service_name = i.split('.nprint')[0]
        print(i)
        nprint_path = nprint_dir+'/'+i
        df = pd.read_csv(nprint_path)
        num_packet = df.shape[0]
        print(df.shape)
        if num_packet != 0:
            try:
                substrings = ['ipv4_src', 'ipv4_dst', 'ipv6_src', 'ipv6_dst','src_ip']

                cols_to_drop = [col for col in df.columns if any(substring in col for substring in substrings)]

                df = df.drop(columns=cols_to_drop)
                cols = df.columns.tolist()
                print(df.shape)
                for col in cols:
                    df[col] = df[col].apply(int_to_rgba)

                output_file = "/home/etbert/netDiffusion/NetDiffusion_Generator/data/preprocessed_fine_tune_imgs/"+service_name+".png"
                dataframe_to_png(df, output_file)
            except:
                print("no_packets")
                continue
