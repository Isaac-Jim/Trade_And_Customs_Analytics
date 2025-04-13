from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('voting_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = ""
    error_message = ""

    if request.method == 'POST':
        try:
            # Get values from form
            fob = float(request.form['fob'])
            cif = float(request.form['cif'])
            mass = float(request.form['mass'])

            # Check if values are greater than 0
            if fob <= 0 or cif <= 0 or mass <= 0:
                error_message = "Values must be greater than 0. Please enter valid positive numbers."
            else:
                # If values are valid, make the prediction
                features = np.array([[fob, cif, mass]])
                prediction = round(model.predict(features)[0], 2)

        except Exception as e:
            error_message = f"Error: {e}"

    # Render the template and pass the error message and prediction
    return render_template('index.html', prediction=prediction, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
