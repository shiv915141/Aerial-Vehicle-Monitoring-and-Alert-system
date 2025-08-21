import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageTk
import requests
import os
import datetime
import tkinter as tk

class AerialVehicleMonitoringSystem:
    def __init__(self, root):
        self.root = root
        self.root.bind('<KeyPress-q>', self.close_window) 
        self.root.protocol("WM_DELETE_WINDOW", self.close_window) 
        self.root.title("Aerial Vehicle Monitoring System")
        self.root.configure(bg='#0B0C10')

        # Title label
        self.title_label = tk.Label(root, text="AERIAL VEHICLE MONITORING SYSTEM", font=("Arial", 20), bg='#0B0C10', fg='#66FCF1')
        self.title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Create a panel to display the video feed
        self.panel = tk.Label(root, bg='#0B0C10')
        self.panel.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Start button
        self.start_button = tk.Button(root, text="Start", font=("Arial", 16), command=self.start_system, bg='#45A29E', fg='#0B0C10')
        self.start_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

        # Load the pre-trained Faster R-CNN model with ResNet-50 backbone
        self.model = fasterrcnn_resnet50_fpn(weights=True)
        self.model.eval()

        # Telegram API details
        self.TOKEN = "6492072912:AAGE9_9gJJnD3_tOEkd7g2ATa0_QNmBw9QE"
        self.CHAT_ID = "5430235789"
        self.base_url = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage"

        # Access the camera
        self.cap = cv2.VideoCapture(0)

        # Variables for video recording
        self.recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = None

        # Threshold and class indices for objects
        self.score_threshold = 0.5
        self.class_labels = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie', 30: 'suitcase', 31: 'frisbee', 32: 'skis', 33: 'snowboard', 34: 'sports ball', 35: 'kite', 36: 'baseball bat', 37: 'baseball glove', 38: 'skateboard', 39: 'surfboard', 40: 'tennis racket', 41: 'bottle', 42: 'wine glass', 43: 'cup', 44: 'fork', 45: 'knife', 46: 'spoon', 47: 'bowl', 48: 'banana', 49: 'apple', 50: 'sandwich', 51: 'orange', 52: 'broccoli', 53: 'carrot', 54: 'hot dog', 55: 'pizza', 56: 'donut', 57: 'cake', 58: 'chair', 59: 'couch', 60: 'potted plant', 61: 'bed', 62: 'dining table', 63: 'toilet', 64: 'tv', 65: 'laptop', 66: 'mouse', 67: 'remote', 68: 'keyboard', 69: 'cell phone', 70: 'microwave', 71: 'oven', 72: 'toaster', 73: 'sink', 74: 'refrigerator', 75: 'book', 76: 'clock', 77: 'vase', 78: 'scissors', 79: 'teddy bear', 80: 'hair drier', 81: 'toothbrush', 82: 'hair brush', 83: 'drone', 84: 'gun', 85: 'shovel', 86: 'wheelchair', 87: 'umbrella'}

        # Define transformations to apply to the input image
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Initialize Telegram message text
        self.telegram_message = "System is online."

        # Send start message
        self.send_telegram_message(self.telegram_message)

    def start_system(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to PIL Image
            pil_img = Image.fromarray(frame_rgb)

            # Apply transformations
            tensor_img = self.transform(pil_img)
            tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension

            # Perform object detection
            with torch.no_grad():
                predictions = self.model(tensor_img)

            # Check if any objects are detected
            objects_detected = False
            for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                if score > self.score_threshold:
                    objects_detected = True
                    class_label = self.class_labels[label.item()]
                    label_text = f'{class_label}: {score:.2f}'
                    # Draw bounding box on the frame
                    box = [int(i) for i in box]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    # Write label text on the frame
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            self.root.update()

            # Break the loop when 'q'
                        # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Send offline message
        self.telegram_message = "System is offline."
        self.send_telegram_message(self.telegram_message)

        # Release the capture and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def close_window(self, event=None):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def send_telegram_message(self, message):
        params = {"chat_id": self.CHAT_ID, "text": message}
        requests.post(self.base_url, params=params)

# Create the main window
root = tk.Tk()
app = AerialVehicleMonitoringSystem(root)
root.mainloop()