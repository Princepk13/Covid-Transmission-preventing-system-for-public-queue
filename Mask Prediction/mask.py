from mtcnn import MTCNN
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing  import image

model = load_model (r"C:\Users\prince\Documents\mask prediction\mask_prediction.h5")
index=['masked','unmasked']
cap = cv2.VideoCapture(0)
detector = MTCNN()


while True:
    ret, frame = cap.read()

    if not ret: break
    frame = cv2.resize(frame, (640, 480))
        
        # Detect face
    img = frame.copy()
    result = detector.detect_faces(img)
    
        # Loop through list (if empty this will be skipped) and overlay green bboxes
        # Format of bboxes is: xmin, ymin (top left), xmax, ymax (bottom right)
    for face in result:
        xmin = face['box'][0]
        ymin = face['box'][1]
        xmax = xmin + face['box'][2]
        ymax = ymin + face['box'][3]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        roi= img[ymin:ymax, xmin:xmax]
        roi=cv2.resize(roi,(64,64))
        roi = roi.astype("float")/ 255.0
        roi = cv2.flip(roi,1,1)
        roi = image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        label=index[np.argmax(model.predict(roi))]
        if label=="unmasked":
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.putText( img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    cv2.imshow("faces", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()