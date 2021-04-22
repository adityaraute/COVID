# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import joblib
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/info.html')
def info():
   return render_template('info.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/uploadpage.html')
def uploadpage():
   return render_template('uploadpage.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')

@app.route('/results.html')
def results():
   return render_template('results.html')

@app.route('/upload_ct.html')
def upload_ct():
   return render_template('upload_ct.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   # resnet_chest = load_model('models/resnet_chest.h5')
   # vgg_chest = load_model('models/vgg_chest.h5')
   # inception_chest = load_model('models/inceptionv3_chest.h5')
   # xception_chest = load_model('models/xception_chest.h5')
   xception = load_model('models/Xception.h5')
   rfc = joblib.load('models/Random_Forest_Classifier.pkl')
   lr = joblib.load('models/Logistic_Regression.pkl')

   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   # resnet_pred = resnet_chest.predict(image)
   # probability = resnet_pred[0]

   params_list = ['Female','Male','Cough','Fever','Fatigue','asthenia','diarrhoea','chest pain','breathing difficulties','hypertension','diabetes','heart disease','lung disease']

   print("Predictions:")
   # if probability[0] > 0.5:
   #    resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')

   # else:
   #    resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')

   # print(resnet_chest_pred)
   # age= request.form['age']
   gender = request.form['gender']
   # location = request.form['location']
   pastCheck = request.form.getlist('pastCheck')
   presentCheck = request.form.getlist('presentCheck')

   symptom_list = []
   if gender=="M":
      symptom_list.extend([0,1])
   else:
      symptom_list.extend([1,0])

   presentCheck_ids = ['presentCheck1', 'presentCheck2', 'presentCheck3', 'presentCheck4', 'presentCheck5', 'presentCheck6', 'presentCheck7']
   pastCheck_ids = ['pastCheck1', 'pastCheck2', 'pastCheck3', 'pastCheck4']

   for x in presentCheck_ids:
      if x in presentCheck:
         symptom_list.append(1)
      else:
         symptom_list.append(0)

   for x in pastCheck_ids:
      if x in pastCheck:
         symptom_list.append(1)
      else:
         symptom_list.append(0)

   xception_pred = xception.predict(image)
   probability_1 = xception_pred[0][0] #todo
   print("------------------------------------------------------------", xception_pred)

   rfc_pred = rfc.predict([symptom_list])
   probability_2 = rfc_pred[0] #todo
   print("===================================================", rfc_pred)

   probability_list = [probability_1, probability_2]

   lr_pred = lr.predict([probability_list])
   final_probability = lr_pred[0]
   print(lr_pred)

   return render_template('results.html', final=final_probability)

def uploaded_chet():
   if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   resnet_chest = load_model('models/resnet_chest.h5')

   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   resnet_pred = resnet_chest.predict(image)
   probability = resnet_pred[0]
   print("Resnet Predictions:")
   if probability[0] > 0.5:
      resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(resnet_chest_pred)
   return render_template('results.html',resnet_chest_pred=resnet_chest_pred)#,vgg_chest_pred=vgg_chest_pred,inception_chest_pred=inception_chest_pred,xception_chest_pred=xception_chest_pred)


if __name__ == '__main__':
   app.secret_key = ".."
   app.run(debug = True)