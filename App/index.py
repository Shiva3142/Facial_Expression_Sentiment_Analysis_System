from Analyzer.ModelProcessor import ModelProcessor,PreprocessInput
from flask import Flask,render_template,redirect,request,jsonify
import cv2
import os
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
inputProcessor=PreprocessInput(face_cascade)
modelProcessor=ModelProcessor()
modelProcessor.setModel("../Model/TransferLearningModel.h5")

App=Flask(__name__)

file_name=""

@App.route("/")
def index():
    return render_template("index.html")


@App.route('/upload_file',methods=['POST','GET'])
def upload():
    global file_name
    if request.method=='POST':
        print(request.files['file'])
        file=request.files['file']
        file_name=file.filename
        extension=file_name.split(".")[-1]
        print(extension)
        file_name=f"input_image.{extension}"
        print(file_name)
        file_path=f"/static/Files/{file_name}"
        file.filename=file_name
        print(file)
        file.save(os.path.dirname(__file__)+f"\\static\\Files\\{file_name}")
        return jsonify({'image_path':file_path})
    else:
        return "error"


@App.route('/get_mage_prediction',methods=['POST','GET'])
def getImagePrediction():
    global file_name
    if request.method=='POST':
        print("Post Method")
        if file_name!="":
            print("File Found")
            image_array=inputProcessor.preprocessTransferLearningInput(f"./static/Files/{file_name}")
            try:
                print(image_array.shape)
                prediction=modelProcessor.predictClass(image_array)
                print(prediction)
                return jsonify({'error':False,"emotion":prediction})
            except:
                return jsonify({'error':True,"emotion":"Image is Not Recognized"})
        else:
            return "ERROR"
    else:
        return "ERROR"

if __name__=="__main__":
    App.run(debug=True,port=5000)