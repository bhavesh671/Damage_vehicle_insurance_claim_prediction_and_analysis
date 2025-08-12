from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

# ==== CONFIGURATION ====
API_KEY = "HnEmzVxYRYHNehhLYlmA"
WORKSPACE = "bhavesh-ofagp"
WORKFLOW_ID = "detect-count-and-visualize-4"

INPUT_FOLDER = "images"  # Folder with your 3000 images
OUTPUT_FOLDER = "labels_yolo_format"  # Where YOLO .txt files will be saved

# ==== SETUP ====
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ==== LOOP THROUGH IMAGES ====
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing {filename}...")

        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": image_path},
                use_cache=True
            )

            img = Image.open(image_path)
            img_width, img_height = img.size

            label_path = os.path.join(
                OUTPUT_FOLDER, os.path.splitext(filename)[0] + ".txt"
            )
            with open(label_path, "w") as f:
                for pred in result["predictions"]["predictions"]:
                    class_id = pred["class_id"]
                    x_center = (pred["x"] + pred["width"] / 2) / img_width
                    y_center = (pred["y"] + pred["height"] / 2) / img_height
                    width = pred["width"] / img_width
                    height = pred["height"] / img_height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
