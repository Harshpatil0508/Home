from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np

# YOLO Model Configuration
def load_yolo_model():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Preprocess Frame for YOLO
def preprocess_frame(frame, net):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net.forward(output_layers)

# Filter Objects Detected by YOLO
def filter_objects(detections, width, height, classes, target_classes=("sofa", "chair", "table")):
    objects = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Detection threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if classes[class_id] in target_classes:
                    objects.append((x, y, w, h, class_id))
    return objects

# Remove Background from Decor Image
def remove_background(decor_item_path):
    image = cv2.imread(decor_item_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load {decor_item_path}. Check the file path.")
    if len(image.shape) < 3 or image.shape[2] != 4:
        raise ValueError("Error: The image must have an alpha channel for transparency.")
    return image

# Overlay Decor Item on Frame
def overlay_virtual_object(frame, x, y, w, h, decor_image):
    decor_resized = cv2.resize(decor_image, (w, h))
    for i in range(h):
        for j in range(w):
            if decor_resized[i, j, 3] != 0:  # Check alpha channel for transparency
                if 0 <= y + i < frame.shape[0] and 0 <= x + j < frame.shape[1]:
                    frame[y + i, x + j] = decor_resized[i, j, :3]
    return frame

# Video Stream Generator
def video_stream():
    net, classes = load_yolo_model()
    decor_image = remove_background("decor_items/chair.png")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        detections = preprocess_frame(frame, net)
        objects = filter_objects(detections, width, height, classes)

        for x, y, w, h, class_id in objects:
            frame = overlay_virtual_object(frame, x, y, w, h, decor_image)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def video_feed(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'ar_app/index.html')
