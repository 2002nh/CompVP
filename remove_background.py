import os
from rembg import remove
from PIL import Image

# Define input and output folders
input_folder = "DatasetFemale"
output_folder = "female_background_black"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Process all images in the folder
for filename in os.listdir(input_folder):

    if filename.lower().endswith(extensions):

        input_path = os.path.join(input_folder, filename)

        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)

        try:
            print(f"Processing: {filename}")

            # Load original image
            input_image = Image.open(input_path).convert("RGBA")

            # Remove original background
            foreground = remove(input_image).convert("RGBA")

            # Create solid black background
            black_background = Image.new(
                "RGBA",
                foreground.size,
                (0, 0, 0, 255)
            )

            # Combine foreground with black background
            output_image = Image.alpha_composite(
                black_background,
                foreground
            )

            # Save final image
            output_image.convert("RGB").save(output_path)

            print(f"Saved: {output_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Background removal completed!")