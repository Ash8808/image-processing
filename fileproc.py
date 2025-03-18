import os
import numpy as np
import cv2
import time
import zipfile
import random  # Import random for randomizing group size

def extract_zip(zip_path, extract_to="extracted_images"):
    """Extracts a zip file to a specified folder."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {extract_to}")
    return extract_to

def process_image(input_path, pixel_array, output_folder):
    """Processes a single image and saves it to the output folder."""
    filename = os.path.basename(input_path)  # Keep original filename
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_processed.jpeg")  

    # Load original image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Skipping {input_path}: File not found or invalid.")
        return
    
    h, w, _ = image.shape
    ph, pw, _ = pixel_array.shape  # Pixel array height and width
    patch_size = ph  # Assuming patches are square (ph x ph)
    
    # Downscale and upscale
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

        for j in range(len(row)):
            if mask[j]:
                current_group.append(j)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        
        if current_group:
            groups.append(current_group)

        # Randomize group size within a range (50 to 150 pixels per group)
        for group in groups:
            max_group_size = random.randint(50, 75)  # Set random max group size
            for start in range(0, len(group), max_group_size):
                sub_group = group[start:start + max_group_size]  # Create sub-groups
                pixel_values = [gray[i, idx] for idx in sub_group]
                pixel_values.sort()  # Sort by luminance
                for k, idx in enumerate(sub_group):
                    gray[i, idx] = pixel_values[k]

    # Assign processed image back to small
    small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Convert to grayscale again
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if pw % patch_size != 0:
        print(f"Skipping {input_path}: Invalid pixel array dimensions.")
        return
    
    num_levels = pw // patch_size  # Number of luminance levels

    # Quantize luminance to 'num_levels' levels
    quantized = np.floor(gray / 255 * num_levels) / num_levels

    # Output image initialization
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Process and replace patches
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            lum_index = int(quantized[i * patch_size, j * patch_size] * num_levels)  # Get luminance level
            lum_index = min(lum_index, num_levels - 1)  # Ensure within range
            patch = pixel_array[:, lum_index * patch_size:(lum_index + 1) * patch_size]

            if patch.shape[1] == 0:
                print(f"Skipping {input_path}: Patch extraction failed.")
                return

            output[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch

    # Save the processed image
    cv2.imwrite(output_path, output)
    print(f"Processed image saved: {output_path}")

def process_folder(input_path, pixel_array_path):
    """Processes all images in a folder (or extracts .zip first) and saves them in a new output folder."""
    
    # If input_path is a zip file, extract it
    if input_path.lower().endswith(".zip"):
        input_folder = extract_zip(input_path)
    else:
        input_folder = input_path  # Use the provided folder

    # Create a unique timestamped folder for each run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = f"processed_images_{timestamp}"
    
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Load pixel array image once
    pixel_array = cv2.imread(pixel_array_path, cv2.IMREAD_COLOR)
    if pixel_array is None:
        raise FileNotFoundError(f"Pixel array '{pixel_array_path}' not found.")

    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Only process image files
            input_path = os.path.join(input_folder, filename)
            process_image(input_path, pixel_array, output_folder)

    print(f"\nAll processed images are saved in: {output_folder}")

# Example use
input_path = "images.zip"  # Can be a folder or a .zip file
pixel_array_path = "Pixel-array-cor.png"  # Your pixel array

process_folder("ezgif-split (5).zip", "Hent.png")
