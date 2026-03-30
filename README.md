# YOLOv11 + CLIP: Natural Language Visual Search

This project takes object detection to the next level by combining **YOLOv11** (for spatial localization) with **CLIP** (for semantic understanding). It allows you to search for specific objects in an image using free-form natural language.

## 🚀 Features
- **Modern Tech Stack:** Uses the latest YOLOv11 and OpenAI's CLIP.
- **Natural Language Querying:** Instead of searching for "car", you can search for "the red sports car with a spoiler".
- **Real-time Feature Extraction:** Automatically crops detected objects and generates semantic embeddings.
- **Visual Output:** Highlights the best-matching object and saves the result as an image.

## 🧠 How it Works
1. **Detection:** YOLOv11 identifies all objects in the scene.
2. **Feature Extraction:** Each detected object is cropped and passed through a CLIP Image Encoder to create a high-dimensional feature vector.
3. **Semantic Matching:** When you enter a text query, it is encoded using CLIP's Text Encoder.
4. **Ranking:** The system calculates the cosine similarity between the text query and all detected objects, selecting the best match.

## 🛠️ Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the system:**
   ```bash
   python main.py
   ```

## 📸 Example Usage
If you have an image with a bus and several people:
- **Query:** "a person wearing a suit"
- **Result:** The system will find the specific person in a suit, even if there are 10 other people in the image.

## 📂 Project Structure
- `main.py`: Core logic for detection and semantic search.
- `requirements.txt`: List of required Python packages.
- `test.jpg`: Input image (auto-downloaded if missing).
- `search_result.jpg`: The output image highlighting your search result.

## 🎯 Future Upgrades
- **Video Support:** Apply this to video streams for real-time tracking of specific descriptions.
- **Vector DB Integration:** Store embeddings in ChromaDB to search across thousands of hours of footage instantly.
- **Web UI:** Add a Gradio or Streamlit interface for a professional feel.
