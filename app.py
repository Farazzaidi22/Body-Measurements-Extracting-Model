from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from predict import preprocess_image, predict_measurements, model

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files or "height" not in request.form:
        return jsonify({"error": "Please provide images and height"}), 400

    images = request.files.getlist("images")
    height = float(request.form["height"])

    results = []
    for img_file in images:
        try:
            # Load the image file and preprocess it
            img = Image.open(img_file).convert("RGB")
            image_tensor = preprocess_image(img)

            # Get predictions
            measurements = predict_measurements(model, image_tensor, height)

            # Convert the predictions to regular Python floats
            result = {
                "filename": img_file.filename,
                "shoulder_length": float(
                    measurements[0]
                ),  # Convert to regular Python float
                "waist": float(measurements[1]),  # Convert to regular Python float
            }

            # Append the result
            results.append(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify(results), 200


if __name__ == "__main__":
    app.run(debug=True)
