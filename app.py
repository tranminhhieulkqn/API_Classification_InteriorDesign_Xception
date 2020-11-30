from flask import Flask, jsonify, render_template, flash, request, redirect

from source.ModelGeneral import ModelGeneral

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

my_model = None

if my_model is None:
    my_model = ModelGeneral()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def home_page():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        try:
            labels = my_model.get_lables()
            xception = my_model.prediction(model="Xception",
                                           image_request=file)
        except():
            return jsonify({
                "success": False,
                "message": "File not exist!"
            }), 404

        return jsonify({
            "success": True,
            "message": "Predicted Results",
            "Labels": labels.tolist(),
            "Xception": [round(num, 2) for num in xception[0].tolist()],
            "Xception Predicted": xception[1]
        }), 200


if __name__ == '__main__':
    app.run()
    if my_model is None:
        my_model = ModelGeneral()
