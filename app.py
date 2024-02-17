from flask import Flask, render_template, request, jsonify
from flask_restful import Resource, Api
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)
api = Api(app)

# Charger le modèle
with open('knn_model.pkl', 'rb') as model_file:
    knn = pickle.load(model_file)

class PredictionResource(Resource):
    def post(self):
        data = request.get_json()
        nouvelles_mesures = [[data['longueur_sepale'], data['largeur_sepale'], data['longueur_petale'], data['largeur_petale']]]
        prediction = knn.predict(nouvelles_mesures)
        return {'prediction': int(prediction[0])}

api.add_resource(PredictionResource, '/predict')

@app.route('/')
def index():
    return render_template('index_api.html')

if __name__ == '__main__':
    app.run(debug=True)
