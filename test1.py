import torch
from models.experimental import attempt_load

# Load the model
model = attempt_load(r'C:\Users\MSI\Documents\project\yolov9\models\detect\yolov9-e.yaml', 'best1.pt')

# Set the model to evaluation mode
model.eval()

# Move the model to the CPU
model.to('cpu')

# Define a function for detection using the camera
import cv2

def detect_camera():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Perform detection on the frame
        with torch.no_grad():
            # Convert the frame to a tensor
            input = torch.from_numpy(frame.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to('cpu')

            # Perform a forward pass through the model
            output = model(input)

            # Post-process the output to get the final detections
            detections = model.postprocess(output, input.shape)[0]

        # Draw the detections on the frame
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
# Call the function to start detection from the camera
detect_camera()