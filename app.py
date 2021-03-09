import pickle

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('LinerRegression_model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('LinerRegression.html')


standard_to = StandardScaler()


@app.route("/Prediction_page", methods=['POST'])
def predict():

    if request.method == 'POST':
        RND = int(request.form['RND'])
        MKT = int(request.form['MKT'])

    prediction = model.predict([[RND, MKT]])
    print(prediction)
    output = prediction[0]

    if output > 0:
        return render_template('LinerRegression.html', prediction_text="You Can Make Profit {} /- RS".format(prediction[0]))
    else:
        return render_template('LinerRegression.html', prediction_text="Sorry you cannot make Profit")

    return render_template('LinerRegression.html')


if __name__ == "__main__":
    app.run(debug=True)
