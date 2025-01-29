import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
from sklearn.decomposition import PCA



app = Flask(__name__) 


model = pickle.load(open('stacking.pkl','rb'))
pca = pickle.load(open('principal.pkl', 'rb'))



@app.route('/')
@app.route('/index') 
def index():
	return render_template('index.html')
@app.route('/login')
def abstract():
	return render_template('about.html')    
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        # df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

 
@app.route('/home')
def home():
    return render_template('test.html')

  
@app.route('/Dataset')
def Dataset():
    return render_template('Dataset.html')




@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = pca.transform(final_features)    
    prediction = model.predict(final_features)
    output = prediction 
    output = ' '.join(output)
    output = output.upper()
    if output == "NORMAL":
        output += "  BEHAVIOUR"
        print(output)
    else:
        # output += "TRAFFIC"
        print(output.upper())
    print(type(output))
    return render_template('test.html', prediction_text= output)
  
    
    
if __name__ == "__main__":
    app.run(debug=True)
