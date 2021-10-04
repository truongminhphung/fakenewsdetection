from flask import Flask, render_template, url_for, request
from utilities import run_detect

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == "POST":
        text = request.form["input_text"]
        print("text: ", text)
        result = run_detect(text)

        return render_template("index.html", result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
