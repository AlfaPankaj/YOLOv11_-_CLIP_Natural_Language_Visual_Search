import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class VisualSearchSystem:
    def __init__(self, yolo_model="yolo11n.pt", clip_model="openai/clip-vit-base-patch32"):
        print(f"Loading YOLO model: {yolo_model}...")
        self.yolo = YOLO(yolo_model)
        
        print(f"Loading CLIP model: {clip_model}...")
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        
        self.detected_objects = []
        self.original_image = None

    def process_image(self, image_path):
        """Detect objects and extract CLIP embeddings."""
        print(f"Processing image: {image_path}...")
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("Could not read image.")
            
        results = self.yolo(self.original_image)
        self.detected_objects = []
        
        pil_image = Image.fromarray(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo.names[cls]
                
                crop = pil_image.crop((x1, y1, x2, y2))
                
                inputs = self.clip_processor(images=crop, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                self.detected_objects.append({
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "label": label,
                    "embedding": image_features.cpu().numpy()
                })
        
        print(f"Detected {len(self.detected_objects)} objects.")

    def search(self, query, output_path="search_result.jpg"):
        """Search for objects matching the natural language query."""
        if not self.detected_objects:
            print("No objects detected. Run process_image first.")
            return

        print(f"Searching for: '{query}'...")
        
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()

        similarities = []
        for obj in self.detected_objects:
            sim = np.dot(obj["embedding"], text_features.T)[0][0]
            similarities.append(sim)

        best_idx = np.argmax(similarities)
        best_match = self.detected_objects[best_idx]
        score = similarities[best_idx]
        
        print(f"Best match found with score: {score:.4f}")
        
        output_img = self.original_image.copy()
        x1, y1, x2, y2 = best_match["box"]
        
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        label_text = f"Query: {query} ({score:.2f})"
        cv2.putText(output_img, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, output_img)
        print(f"Result saved to {output_path}")

if __name__ == "__main__":
    import os
    import requests

    system = VisualSearchSystem()
    
    if not os.path.exists("test.jpg"):
        print("Downloading sample image...")
        sample_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
        response = requests.get(sample_url)
        with open("test.jpg", "wb") as f:
            f.write(response.content)
        print("Sample image 'test.jpg' downloaded.")

    system.process_image("test.jpg")
    
    print("\n--- Visual Search System Ready ---")
    print("Try queries like: 'person in a suit', 'blue bus', 'the wheel', 'a man with a backpack'")
    while True:
        query = input("\nEnter search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        system.search(query)
