from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        results = model(image_path)
        for r in results:
            annotated = r.plot()
            cv2.imwrite(image_path, annotated)

        return render_template("index.html", image_path=image_path)
    
    return render_template("index.html", image_path=None)

if __name__ == "__main__":
    app.run(debug=True)