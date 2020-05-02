import numpy as np
import cv2
import pickle
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read("trainner.yml")
label = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        id_, conf = recogniser.predict(roi_gray)
        if conf<=85:
            print(conf)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            colour = (255,255,255)
            stroke = 2
            cv2.putText(frame, name,(x,y), font, 0.5, colour, stroke, cv2.LINE_AA)
        elif conf >= 100:
            print(conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Other"
            colour = (255,255,255)
            stroke = 2
            cv2.putText(frame, name,(x,y), font, 0.5, colour, stroke, cv2.LINE_AA)
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_frame)
        colour = (255,0,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), colour, stroke)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()