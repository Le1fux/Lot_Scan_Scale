# main_app.py
# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("../docs/smartlot_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).resize((128,128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return jsonify({'prediction': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
