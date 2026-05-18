from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

input_path = Path(r"C:\Users\ROG Zephyrus G15\Downloads\IMG_8343 (1).HEIC")
output_folder = Path(r"C:\Users\ROG Zephyrus G15\Documents\AI\raw_dataset\circle\converted")

output_folder.mkdir(parents=True, exist_ok=True)

# Supported image formats to convert to JPG
supported_formats = {".heic", ".png", ".jpg", ".jpeg"}

if input_path.is_file():
    files_to_convert = [input_path]
elif input_path.is_dir():
    files_to_convert = [path for path in input_path.iterdir() if path.suffix.lower() in supported_formats]
else:
    raise FileNotFoundError(f"Input path does not exist: {input_path}")

for file_path in files_to_convert:
    try:
        img = Image.open(file_path)

        # Convert RGBA to RGB if necessary for formats with transparency.
        if img.mode in ("RGBA", "LA", "P"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = rgb_img

        new_name = f"{file_path.stem}.jpg"
        output_path = output_folder / new_name
        img.save(output_path, "JPEG", quality=95)
        print(f"Converted: {file_path.name} -> {new_name}")
    except Exception as e:
        print(f"Error converting {file_path.name}: {e}")

print("Conversion completed!")