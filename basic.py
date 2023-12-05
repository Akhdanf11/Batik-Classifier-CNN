import numpy as np
import tensorflow as tf

from flask import Flask, request
from PIL import  Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/batik', methods = ['POST'])

def batik_classifier():
    #Ambil gambar yang dikirim pas request
    image_request = request.files['image']

    #konversi gambar menjadi array
    image_pil = Image.open(image_request)

    #ngeresize gambar
    expected_size = (256,256)
    resized_image_pil = image_pil.resize(expected_size)

    #generate array dengan numpy
    image_array = np.array(resized_image_pil)
    rescaled_image_array = image_array/255.
    batched_rescaled_image_array = np.array([rescaled_image_array])
    #print(batched_rescaled_image_array.shape)

    #load model
    loaded_model = tf.keras.models.load_model("batik-classifier.h5")
    #print(loaded_model.get_config())
    result = loaded_model.predict(batched_rescaled_image_array)
    get_formated_predict_result(result)

    #[
    # [2.8887406e-32 1.0000000e+00 6.9345270e-36 2.7392862e-24 0.0000000e+00
    #  0.0000000e+00 0.0000000e+00 0.0000000e+00 1.5596536e-19 1.4313492e-27]
    # ]

    return get_formated_predict_result(result)

def get_formated_predict_result(predict_result) :
    class_indices = {'Batik Cap Asem Arang': 0, 'Batik Cap Asem Sinom': 1, 'Batik Cap Asem Warak': 2, 'Batik Cap Blekok': 3, 'Batik Cap Blekok Warak': 4, 'Batik Cap Gambang Semarangan': 5, 'Batik Cap Kembang Sepatu': 6, 'Batik Cap Semarangan': 7, 'Batik Cap Tugu Muda': 8, 'Batik Cap Warak Beras Utah': 9}
    inverted_class_indices = {}

    for key in class_indices:
        class_indices_key = key
        class_indices_value = class_indices[key]
        inverted_class_indices[class_indices_value] = class_indices_key

    processed_predict_result = predict_result[0]
    maxIndex = 0
    maxValue = 0

    for index in range(len(processed_predict_result)):
        if processed_predict_result[index] > maxValue:
            maxValue = processed_predict_result[index]
            maxIndex = index

    return inverted_class_indices[maxIndex]

if __name__ == "__main__":
    app.run()

