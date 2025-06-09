from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Load your test image
image_path = "your_test_image.jpg"  # change to your test image
results = model(image_path)

# Show the result with boxes
for r in results:
    annotated = r.plot()
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()