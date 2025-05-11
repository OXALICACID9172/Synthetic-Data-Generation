from PIL import Image
import numpy as np
import os

def process_image(input_file, output_file):
    # Open and convert to RGBA
    image = Image.open(input_file).convert('RGBA')
    # Convert to numpy array
    img_array = np.array(image, dtype=np.uint8)
    
    # Threshold values
    r_red, g_red, b_red = 0, 160, 100
    r_green, g_green, b_green = 0, 160, 100
    r_blue, g_blue, b_blue = 0, 160, 100
    
    # Create masks for each condition
    red_mask = (img_array[:,:,0] > r_red) & (img_array[:,:,1] < g_red) & (img_array[:,:,2] < b_red)
    green_mask = (img_array[:,:,0] < r_green) & (img_array[:,:,1] > g_green) & (img_array[:,:,2] < b_green)
    blue_mask = (img_array[:,:,0] < r_blue) & (img_array[:,:,1] < g_blue) & (img_array[:,:,2] > b_blue)
    
    # Default case: find max color
    max_colors = np.argmax(img_array[:,:,:3], axis=2)
    
    # Create output array
    output = np.zeros_like(img_array)
    output[:,:,3] = 255  # Set alpha channel
    
    # Apply colors based on masks
    output[red_mask] = [255, 0, 0, 255]
    output[green_mask] = [0, 255, 0, 255]
    output[blue_mask] = [0, 0, 255, 255]
    
    # Apply max color where no mask matched
    remaining = ~(red_mask | green_mask | blue_mask)
    output[remaining & (max_colors == 0)] = [255, 0, 0, 255]
    output[remaining & (max_colors == 1)] = [0, 255, 0, 255]
    output[remaining & (max_colors == 2)] = [0, 0, 255, 255]
    
    # Convert back to image and save
    Image.fromarray(output).save(output_file)

# Your existing loop
org_dir = '../data/generated_imgs/'
counter = 0
for i in os.listdir(org_dir):
    if 'png' in i:
        input_file = os.path.join(org_dir, i)
        output_file = os.path.join('../data/color_processed_generated_imgs/', i)
        process_image(input_file, output_file)
        counter += 1
        print(counter)
