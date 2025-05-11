import pandas as pd
from PIL import Image
import numpy as np
import os

# Existing helper functions (unchanged)
def ip_to_binary(ip_address):
    octets = ip_address.split(".")
    binary_octets = [bin(int(octet))[2:].zfill(8) for octet in octets]
    return "".join(binary_octets)

def binary_to_ip(binary_ip_address):
    if len(binary_ip_address) != 32:
        raise ValueError("Input binary string must be 32 bits")
    octets = [binary_ip_address[i:i+8] for i in range(0, 32, 8)]
    return ".".join(str(int(octet, 2)) for octet in octets)

def rgba_to_ip(rgba):
    return '.'.join(map(str, rgba))

def int_to_rgba(A):
    if A == 1:
        return (255, 0, 0, 255)
    elif A == 0:
        return (0, 255, 0, 255)
    elif A == -1:
        return (0, 0, 255, 255)
    elif A > 1:
        return (255, 0, 0, A)
    elif A < -1:
        return (0, 0, 255, abs(A))
    return None

def rgba_to_int(rgba):
    if rgba == (255, 0, 0, 255):
        return 1
    elif rgba == (0, 255, 0, 255):
        return 0
    elif rgba == (0, 0, 255, 255):
        return -1
    elif rgba[0] == 255 and rgba[1] == 0 and rgba[2] == 0:
        return rgba[3]
    elif rgba[0] == 0 and rgba[1] == 0 and rgba[2] == 255:
        return -rgba[3]
    return None

# Optimized png_to_dataframe using NumPy
def png_to_dataframe(input_file):
    # Open image and convert to numpy array
    img = Image.open(input_file)
    img_array = np.array(img, dtype=np.uint8)  # Shape: (height, width, 4)
    
    # Vectorized conversion of RGBA to integers
    height, width, _ = img_array.shape
    result = np.zeros((height, width), dtype=int)
    
    # Masks for each condition
    red_full = (img_array[:,:,0] == 255) & (img_array[:,:,1] == 0) & (img_array[:,:,2] == 0) & (img_array[:,:,3] == 255)
    green_full = (img_array[:,:,0] == 0) & (img_array[:,:,1] == 255) & (img_array[:,:,2] == 0) & (img_array[:,:,3] == 255)
    blue_full = (img_array[:,:,0] == 0) & (img_array[:,:,1] == 0) & (img_array[:,:,2] == 255) & (img_array[:,:,3] == 255)
    red_var = (img_array[:,:,0] == 255) & (img_array[:,:,1] == 0) & (img_array[:,:,2] == 0) & (img_array[:,:,3] != 255)
    blue_var = (img_array[:,:,0] == 0) & (img_array[:,:,1] == 0) & (img_array[:,:,2] == 255) & (img_array[:,:,3] != 255)
    
    # Assign values based on masks
    result[red_full] = 1
    result[green_full] = 0
    result[blue_full] = -1
    result[red_var] = img_array[red_var, 3]  # Variable alpha for red
    result[blue_var] = -img_array[blue_var, 3]  # Variable alpha for blue
    
    # Create DataFrame directly from numpy array
    return pd.DataFrame(result, columns=cols)

# Load original DataFrame for columns
org_nprint = 'column_example.nprint'
org_df = pd.read_csv(org_nprint).drop('Unnamed: 0', axis=1)
cols = org_df.columns.tolist()

# Process PNG files with optimized function
org_dir = '../data/color_processed_generated_imgs/'
counter=0
for i in os.listdir(org_dir):
    if 'png' in i:
        input_file = os.path.join(org_dir, i)  # Use os.path.join for safer path handling
        reverse_df = png_to_dataframe(input_file)
        output_file = os.path.join('../data/generated_nprint/', i.replace('png', 'nprint'))
        reverse_df.to_csv(output_file)
        counter+=1
        print(counter)
