from flask import Flask, request , jsonify
import model.predictor as predictor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def predict():
        data = request.json
        input_data = data['input'] 
        prediction = predictor.predictToUser(input_data)
        return jsonify({'prediction': prediction})

if __name__ == 'main':
    app.run(port=3000)
