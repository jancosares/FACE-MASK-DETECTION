from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    original_path = None
    detected_path = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file and file.filename:
            filename = file.filename

            # Save original image
            original_path = os.path.join("static", "original_" + filename)
            file.save(original_path)

            # Run detection and save annotated result
            results = model(original_path)
            for r in results:
                annotated = r.plot()

                # Save detected image separately
                detected_path = os.path.join("static", "detected_" + filename)
                cv2.imwrite(detected_path, annotated)

    return render_template(
        "index.html",
        original_path=original_path,
        detected_path=detected_path,
        filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os

# app = Flask(__name__)
# model = YOLO("best.pt")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["image"]
#         image_path = os.path.join("static", file.filename)
#         file.save(image_path)

#         results = model(image_path)
#         for r in results:
#             annotated = r.plot()
#             cv2.imwrite(image_path, annotated)

#         return render_template("index.html", image_path=image_path, filename=filename)
    
#     return render_template("index.html", image_path=None)

"""from flask import Flask, request, render_template
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
"""