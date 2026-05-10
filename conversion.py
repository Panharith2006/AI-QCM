from PIL import Image
import pillow_heif
import os

pillow_heif.register_heif_opener()

input_folder = r"D:\Year 3\AI\raw_dataset\circle"
output_folder = r"D:\Year 3\AI\raw_dataset\circle\converted"

os.makedirs(output_folder, exist_ok=True)

# Supported image formats to convert to JPG
supported_formats = (".HEIC", ".heic", ".PNG", ".png", ".jpg", ".jpeg", ".JPG", ".JPEG")

for file in os.listdir(input_folder):
    if file.endswith(supported_formats):
        try:
            img = Image.open(os.path.join(input_folder, file))
            
            # Convert RGBA to RGB if necessary (for PNG with transparency)
            if img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = rgb_img
            
            new_name = os.path.splitext(file)[0] + ".jpg"
            img.save(os.path.join(output_folder, new_name), "JPEG", quality=95)
            print(f"Converted: {file} → {new_name}")
        except Exception as e:
            print(f"Error converting {file}: {e}")

print("Conversion completed!")