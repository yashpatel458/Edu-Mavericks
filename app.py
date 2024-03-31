from flask import Flask, render_template, redirect, url_for
import subprocess
from pymongo import MongoClient

completeData = []
def readData():
    # Connect to MongoDB
    client = MongoClient('mongodb://127.0.0.1:27017/')
    db = client['AdtProject']
    collection = db['education']

    # Get all data from the collection
    data = collection.find()

    # Iterate over the data and print each document
    for document in data:
        completeData.append(document)

readData()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/donoothing')
def donoothing():
    return render_template('close.html')

@app.route('/link1')
def link1():
    subprocess.Popen(['streamlit', 'run', 'education.py'])
    return redirect(url_for('donoothing'))

@app.route('/link2')
def link2():
    subprocess.Popen(['streamlit', 'run', 'wagesEducation.py'])
    return redirect(url_for('donoothing'))

@app.route('/link3')
def link3():
    subprocess.Popen(['streamlit', 'run', 'employementForecast.py'])
    return redirect(url_for('donoothing'))

if __name__ == '__main__':
    app.run(debug=True)
