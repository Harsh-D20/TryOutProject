from diffusers import StableDiffusionPipeline
import spacy
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

IMG_NAME = "generated_image_4.png"
PROMPT = "A boy and a girl sitting on a bench"

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

def generate_image(prompt, name):
    image = pipe(prompt).images[0]
    image.save(name)

def get_detected_objects():
    # Load YOLOv8 model
    model = YOLO("yolov8x.pt")  # Use a pre-trained YOLOv8 model

    # Run object detection
    objects = model(IMG_NAME)

    detections = []
    for obj in objects:
        for box in obj.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score
            label = model.names[cls]  # Object name
            detections.append((label, x1, y1, x2, y2, conf))

    # Print detected objects and coordinates
    for obj in detections:
        print(f"Object: {obj[0]}, Coordinates: ({obj[1]}, {obj[2]}, {obj[3]}, {obj[4]}), Confidence: {obj[5]}")
    
    return detections

def get_prompt_nouns():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(PROMPT)

    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    print("Nouns in Prompt:", nouns)
    return nouns 

def get_matching_objects(prompt_nouns, found):
    matched_objects = [obj for obj in found if obj[0] in prompt_nouns]
    print("Matched Objects:", matched_objects)
    return matched_objects

def show_image(name, objects):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

    # Draw bounding boxes
    for obj in objects:
        label, x1, y1, x2, y2, conf = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

        # Add label text
        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    generate_image(PROMPT, IMG_NAME)
    found = get_detected_objects()
    nouns = get_prompt_nouns()
    matches = get_matching_objects(nouns, found)
    show_image(IMG_NAME, found)