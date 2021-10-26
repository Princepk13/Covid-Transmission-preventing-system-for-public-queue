import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing  import image


model = load_model (r"C:\Users\prince\Documents\mask prediction\mask_prediction.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
index=['masked','unmasked']
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
   
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi= gray[y:y+h, x:x+w]
        roi=cv2.resize(roi,(64,64))
        roi = roi.astype("float")/ 255.0
        roi = cv2.flip(roi,1,1)
        roi = image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        label=index[np.argmax(model.predict(roi))]
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()