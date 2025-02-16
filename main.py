from diffusers import StableDiffusionPipeline
import spacy
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

IMG_NAME = "generated_image_4.png"
PROMPT = "A boy and a girl sitting on a bench"

# get stable diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# generates image using prompt and stable diffusion and saves it with file name 'name'
def generate_image(prompt, name):
    image = pipe(prompt).images[0]
    image.save(name)

# use YOLO model to detect all objects in the image given 
def get_detected_objects(name):
    model = YOLO("yolov8x.pt") 

    # gets objects
    objects = model(name)

    # get xy coordinates of each object, conf level, and label
    detections = []
    for obj in objects:
        for box in obj.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            detections.append((label, x1, y1, x2, y2, conf))

    for obj in detections:
        print(f"Object: {obj[0]}, Coordinates: ({obj[1]}, {obj[2]}, {obj[3]}, {obj[4]}), Confidence: {obj[5]}")
    
    return detections

# gets all nouns used in the prompt, and returns a list of them
def get_prompt_nouns():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(PROMPT)

    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    print("Nouns in Prompt:", nouns)
    return nouns 

# takes labeled image objects and prompt nouns and returns a list of their intersection
def get_matching_objects(prompt_nouns, found):
    matched_objects = [obj for obj in found if obj[0] in prompt_nouns]
    print("Matched Objects:", matched_objects)
    return matched_objects

# displays image with bounding boxes for each detected object
def show_image(name, objects):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for obj in objects:
        label, x1, y1, x2, y2, conf = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# driver code
if __name__ == "__main__":
    generate_image(PROMPT, IMG_NAME)
    found = get_detected_objects()
    nouns = get_prompt_nouns()
    matches = get_matching_objects(nouns, found)
    show_image(IMG_NAME, found)