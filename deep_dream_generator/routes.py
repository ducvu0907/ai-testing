import os
from flask import Flask, request,   render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from transform import deepdream
app = Flask(__name__)

@app.route('/', methods=["GET"])
def default():
  return render_template("index.html")

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"jpg", "jpeg", "png"}

@app.route("/transform", methods=["POST"])
def upload_file():
  if "image" not in request.files:
    return jsonify({"error": "no file part"})

  image = request.files["image"]
  if not image.filename:
    return jsonify({"error": "no file selected"})

  if image and allowed_file(image.filename):
    filename = secure_filename(image.filename)
    filepath = os.path.join("uploads", filename)
    image.save(filepath)
    result = deepdream(filepath)