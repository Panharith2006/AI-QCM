# setup_models.py
import easyocr
import os

print("Downloading EasyOCR models...")
reader = easyocr.Reader(['en'], gpu=True, verbose=True)
print("✅ Models ready! You can now run: streamlit run app/streamlit_app.py")