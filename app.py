from flask import *
from DL_model.extract_fg import ExtractFG

import os

INPUT_IMAGE_PATH = 'uploads\input.png'
OUTPUT_IMAGE_PATH = 'uploads\output.png'

app = Flask(__name__)


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/')
def home():
    if os.path.isfile(OUTPUT_IMAGE_PATH):
        os.remove(OUTPUT_IMAGE_PATH)

    if os.path.isfile(INPUT_IMAGE_PATH):
        os.remove(INPUT_IMAGE_PATH)

    return render_template('home.html')


def get_extracted_image():
    extract_fg = ExtractFG(input_image_path=INPUT_IMAGE_PATH,
                           output_image_path=OUTPUT_IMAGE_PATH)

    extract_fg.getExtractedImage()


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(INPUT_IMAGE_PATH)
            get_extracted_image()
            return render_template('upload.html')
        else:
            print('Upload Image you DumbA$$')
    return render_template('home.html')


@app.route('/output_image')
def output_image():
    return send_from_directory('uploads', 'output.png')


@app.route('/download')
def download_files():
    return send_file(OUTPUT_IMAGE_PATH, as_attachment=True)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
