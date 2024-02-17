from flask import Flask, request, render_template
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load the dataset and train the model
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get the data from the form
        longueur_sepale = float(request.form['longueur_sepale'])
        largeur_sepale = float(request.form['largeur_sepale'])
        longueur_petale = float(request.form['longueur_petale'])
        largeur_petale = float(request.form['largeur_petale'])

        # Validate the input values
        if not (0 <= longueur_sepale <= 10 and 0 <= largeur_sepale <= 10 and 0 <= longueur_petale <= 10 and 0 <= largeur_petale <= 10):
            raise ValueError("Les mesures doivent être comprises entre 0 et 10.")

        # Make the prediction
        nouvelles_mesures = np.array([[longueur_sepale, largeur_sepale, longueur_petale, largeur_petale]])
        predictions = knn.predict(nouvelles_mesures)
        prediction_proba = knn.predict_proba(nouvelles_mesures)

        # Render the result template with the prediction and probabilities
        return render_template('result.html', prediction=predictions[0], proba_setosa=prediction_proba[0][0], proba_versicolor=prediction_proba[0][1], proba_virginica=prediction_proba[0][2])

    except ValueError as e:
        # Handle validation errors
        return render_template('result.html', error_message=str(e))

if __name__ == '__main__':
    app.run(host="localhost", debug=True)
