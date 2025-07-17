import gzip
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 🔒 Use env variable in production

# Load gzipped model
with gzip.open('mymodel1.pkl.gz', 'rb') as f:
    model = pickle.load(f)

# Dummy login credentials
users = {'admin': 'password123'}

@app.route('/')
def home():
    if 'username' in session:
        return f"Welcome, {session['username']}!"
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pword = request.form['password']
        if users.get(uname) == pword:
            session['username'] = uname
            return redirect(url_for('home'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    input_data = np.array([data['features']])
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
