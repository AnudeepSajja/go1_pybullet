import os
import shutil
import re

# Define the source directory containing the original images
source_dir = '/home/anudeep/devel/workspace/src/data/images'
# Define the target directory where you want to save the new images
target_dir = '/home/anudeep/devel/workspace/src/data/images_processed'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get all image files in the source directory
image_files = [f for f in os.listdir(source_dir) if re.match(r'image_\d+\.png', f)]

# Extract the highest index from the image names
if image_files:
    max_index = max(int(re.search(r'\d+', f).group()) for f in image_files)
else:
    max_index = 0

num_images = max_index + 1

batch = 100

print(f"Processing {num_images} images in batches of {batch}...")

# Loop through each batch of images
for i in range(0, num_images, batch):
    # Determine the source image file name for the first image in the next batch
    source_image_name = f'image_{i + 100}.png'
    
    # Construct full path to the source image
    source_image_path = os.path.join(source_dir, source_image_name)
    
    # Check if the source image exists
    if os.path.exists(source_image_path):
        # Loop through each index in the current batch (0-99)
        for j in range(100):
            # Calculate the corresponding image index
            new_image_index = i + j+1
            
            # Construct new image name
            new_image_name = f'image_{new_image_index}.png'
            
            # Construct full path for the new image in the target directory
            target_image_path = os.path.join(target_dir, new_image_name)
            
            # Copy the source image to the target path
            shutil.copy(source_image_path, target_image_path)

print("Images have been processed and saved successfully!")
