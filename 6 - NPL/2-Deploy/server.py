import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
model = joblib.load('../../6 - NPL files/model_svm.joblib')
model2 = load_model('../../6 - NPL files/keras_lstm.h5')
model2_tokenizer = pickle.load(open('../../6 - NPL files/keras_lstm_tokenizer','br')) 
#engin_mode = 'dl'

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    model_engine = int_features[1]
   
    if model_engine == 'ml':

        # final_features = [np.array(int_features)]
        # prediction = model.predict(['This is terrible place \
        #                            disgusting unfriendly'])
        prediction = model.predict([int_features[0]])

    elif model_engine == 'dl':
        test_encoded = model2_tokenizer.texts_to_sequences(int_features[0])
        test_padded = pad_sequences(test_encoded, maxlen=100, padding='post')
        result =  model2.predict_classes(test_padded)
        prediction = [result.tolist(),type(result)]
        dict_map = {1:5,0:1}
        prediction = list(map(dict_map.get, result.tolist()[0]))[0]
    output = int(prediction)
 
    return render_template('index.html',
                           prediction_text='The rating \
                           of this review is {}'.format(output))

@app.route('/results',methods=['GET', 'POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict(['This is terrible place disgusting unfriendly'])
    # return 'The rating based on the text probably is {}'.format(prediction)
    return jsonify(int(prediction))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
