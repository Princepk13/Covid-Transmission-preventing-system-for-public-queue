from flask import Flask,render_template,Response
import numpy as np
import imutils
import time
import cv2
import os
import math
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from itertools import chain 
from constants import *

app=Flask(__name__)

video = cv2.VideoCapture(0)

prototxtPath = "deploy.prototxt.txt"
weightsPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

model = load_model("mask_prediction.h5")
index=["masked","Unmasked"]

LABELS = open(YOLOV3_LABELS_PATH).read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')


neural_net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG_PATH, YOLOV3_WEIGHTS_PATH)
layer_names = neural_net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in neural_net.getUnconnectedOutLayers()]
writer = None
(W,H)=(None,None)


@app.route('/')
def index():
	return render_template('main.html')

@app.route('/video_feed1')
def video_feed1():
	return Response(gen_frames1(),mimetype='multipart/x-mixed-replace;boundary=frame')
	
def gen_frames1():
    while True:
        _, frame1 = video.read()
        if frame1 is None:
            break
        frame1=cv2.resize(frame1,(640,480))    
        h,w = frame1.shape[:2]
    
        blob = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
   
        net.setInput(blob)
        detections = net.forward()
    
        for i in range(0 , detections.shape[2]):
            confidence = detections[0, 0, i,2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x,y,u,v) = box.astype("int")
                cv2.rectangle(frame1,(x,y),(u,v),(0,255,0),2)
            
                crop = frame1[y:v, x:u]
                if crop is None:
                    continue
                roi=cv2.resize(crop,(64,64))
                roi = roi.astype("float")/ 255.0
                roi = cv2.flip(roi,1,1)
                img_array = img_to_array(roi)
                face = np.expand_dims(img_array, axis=0)
                ind=np.argmax(model.predict(face))
                if ind==0:
                    label="masked"
                else:
                    label="unmasked"
                        
               
                color = (0,255,0) if label == "masked" else (0,0,255)
			
                cv2.putText(frame1, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,
                        color,2)
                cv2.rectangle(frame1,(x,y),(u,v),color,2)
        ret,jpeg=cv2.imencode('.jpg',frame1)
        frame1=jpeg.tobytes()
        yield(b'--frame1\r\n'
              b'Content-Type:image/jpeg\r\n\r\n'+frame1+b'\r\n\r\n')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(),mimetype='multipart/x-mixed-replace;boundary=frame')

def gen_frames2():
    (W,H)=(None,None)
    while True:
        (grabbed, frame2) = video.read()

        if not grabbed:
            break
        if frame2 is None:
            break

        
        if W is None or H is None:
            H, W = (frame2.shape[0], frame2.shape[1])

        blob1 = cv2.dnn.blobFromImage(frame2, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        neural_net.setInput(blob1)

        start_time = time.time()
        layer_outputs = neural_net.forward(layer_names)
        end_time = time.time()
    
        boxes = []
        confidences = []
        classIDs = []
        lines = []
        box_centers = []

        for output in layer_outputs:
            for detection1 in output:
            
                scores = detection1[5:]
                classID = np.argmax(scores)
                confidence1 = scores[classID]
            
                if confidence1 > 0.5 and classID == 0:
                    box = detection1[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype('int')
                
                    a = int(centerX - (width / 2))
                    b = int(centerY - (height / 2))
                
                    box_centers = [centerX, centerY]

                    boxes.append([a, b, int(width), int(height)])
                    confidences.append(float(confidence1))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
        if len(idxs) > 0:
            unsafe = []
            count = 0
        
            for i in idxs.flatten():
            
                (a, b) = (boxes[i][0], boxes[i][1])
                (m, n) = (boxes[i][2], boxes[i][3])
                centeriX = boxes[i][0] + (boxes[i][2] // 2)
                centeriY = boxes[i][1] + (boxes[i][3] // 2)

                color = [int(c) for c in COLORS[classIDs[i]]]
                text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])

                idxs_copy = list(idxs.flatten())
                idxs_copy.remove(i)

                for j in np.array(idxs_copy):
                    centerjX = boxes[j][0] + (boxes[j][2] // 2)
                    centerjY = boxes[j][1] + (boxes[j][3] // 2)

                    distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))

                    if distance <= SAFE_DISTANCE:
                        cv2.line(frame2, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                        unsafe.append([centerjX, centerjY])
                        unsafe.append([centeriX, centeriY])

                if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                    count += 1
                    cv2.rectangle(frame2, (a, b), (a + m, b + n), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame2, (a, b), (a + m, b + n), (0, 255, 0), 2)

                cv2.putText(frame2, text, (a, b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame2, (50, 50), (450, 90), (0, 0, 0), -1)
                image1=cv2.putText(frame2, 'No. of people unsafe: {}'.format(count), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
        
    
        
        ret1,jpeg1=cv2.imencode('.jpg',frame2)
        frame2=jpeg1.tobytes()
        yield(b'--frame2\r\n'
              b'Content-Type:image/jpeg\r\n\r\n'+frame2+b'\r\n\r\n')


    
    
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
cv2.destroyAllWindows()
