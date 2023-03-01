from flask import Flask,render_template,request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



app=Flask(__name__)
model=load_model('ECG.h5')
@app.route("/about")
def home():
	return render_template("index.html")

@app.route("/")
def about():
	return render_template("index.html")

@app.route("/info")
def info():
	return render_template("info.html")

@app.route("/upload")
def test():
	return render_template("predict.html")

@app.route("/predict",methods=["GET","POST"])
def upload():
	if request.method=='POST':
		f=request.files['file']
		basepath=os.path.dirname('__file__')
		filepath=os.path.join(basepath,"uploads",f.filename)
		f.save(filepath)
		
		img=image.load_img(filepath,target_size=(64,64)) 
		x=image.img_to_array(img)
		x=np.expand_dims(x,axis=0)
		pred=model.predict(x) 
		y_pred = np.argmax(pred)
		print("prediction",y_pred)


		index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction','Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
		result=str(index[y_pred])
		return result

	return None

if __name__=="__main__":
 app.run(debug=True) 
