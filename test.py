import numpy as np
import cv2
import time
import random
 
def process_image(input_path, pixel_array_path):
    # Generate filename 
    timestamp = int(time.time()) 
    output_path = f"output_{timestamp}.jpeg"  # Create a new filename
 
    # Load original image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Input image '{input_path}' not found.")
    h, w, _ = image.shape
    # Downscale and upscale
    pixil_array = cv2.imread(pixel_array_path, cv2.IMREAD_COLOR)
    ph, pw, _ = pixil_array.shape
    small = cv2.resize(image, (w // ph, h // ph), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Define threshold values
    lower_thresh, upper_thresh = 75, 250  # Adjust as needed

    # Process each row independently
    for i in range(gray.shape[0]):
        row = gray[i, :]
        mask = (row >= lower_thresh) & (row <= upper_thresh)

        # Identify groups of uninterrupted pixels in threshold range
        groups = []
        current_group = []
        
        # Set a randomized max group size per row (±5 variation)
        max_group_size = 100 + random.randint(-5, 5)

        for j in range(len(row)):
            if mask[j]:
                current_group.append(j)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        
        if current_group:
            groups.append(current_group)

        # Limit group size and sort each separately
        for group in groups:
            for start in range(0, len(group), max_group_size):
                sub_group = group[start:start + max_group_size]  # Create sub-groups with randomized max size
                pixel_values = [gray[i, idx] for idx in sub_group]
                pixel_values.sort()  # Sort by luminance
                for k, idx in enumerate(sub_group):
                    gray[i, idx] = pixel_values[k]

    # Assign processed image back to small
    small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert to grayscale again
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pixel array image
    pixil_array = cv2.imread(pixel_array_path, cv2.IMREAD_COLOR)
    
    # Dynamically determine patch size and number of luminance levels
    patch_size = ph  # Assuming patches are square (ph x ph)

    if pw % patch_size != 0:
        raise ValueError(f"Pixil array width ({pw}) must be a multiple of its height ({ph}).")
    
    num_levels = pw // patch_size  # Number of luminance levels
    
    if num_levels < 1:
        raise ValueError("Invalid pixil array: width must allow at least one luminance level.")

    # Quantize luminance to 'num_levels' levels
    quantized = np.floor(gray / 255 * num_levels) / num_levels

    # Output image initialization
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Process and replace patches
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            lum_index = int(quantized[i * patch_size, j * patch_size] * num_levels)  # Get luminance level
            
            # Ensure index is within valid range
            lum_index = min(lum_index, num_levels - 1)

            # Extract patch safely
            patch = pixil_array[:, lum_index * patch_size:(lum_index + 1) * patch_size]

            if patch.shape[1] == 0:  # Check if width is 0 (invalid slice)
                raise ValueError(f"Patch extraction failed for lum_index {lum_index}, patch_size {patch_size}")

            # Assign patch to output
            output[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch
    # Save the new image to generated filename
    cv2.imwrite(output_path, output)
    print(f"Processed image saved as {output_path}")
 
# use
process_image("Shiz.jpg", "Trans-Pixel.png")
