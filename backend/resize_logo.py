from PIL import Image
import os

# Path to your original logo
original_path = r"C:\Users\Ravs\OneDrive\Desktop\medical-ai-platform\frontend\src\assets\takku-logo.png"
output_path = r"C:\Users\Ravs\OneDrive\Desktop\medical-ai-platform\frontend\src\assets\takku-logo-resized.png"

try:
    # Open and resize image
    with Image.open(original_path) as img:
        # Resize to 200x200 pixels
        resized_img = img.resize((200, 200), Image.Resampling.LANCZOS)
        
        # Save optimized version
        resized_img.save(output_path, "PNG", optimize=True, quality=85)
        
    print(f"✅ Logo resized successfully!")
    print(f"Original: {os.path.getsize(original_path)} bytes")
    print(f"Resized: {os.path.getsize(output_path)} bytes")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure PIL is installed: pip install Pillow")