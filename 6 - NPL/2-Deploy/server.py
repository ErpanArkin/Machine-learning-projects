import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_svm.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]

    # final_features = [np.array(int_features)]
    # prediction = model.predict(['This is terrible place \
    #                            disgusting unfriendly'])
    prediction = model.predict(int_features)

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
