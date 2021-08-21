import cv2
import os
import imutils

nombrePersona = 'Ejemplo1'

dataPath = 'C:/Users/iUser/Documents/GitHub/biometrico/Data'
personaPath = dataPath + '/' + nombrePersona
#print(personaPath)

if not os.path.exists(personaPath):
    print('Carpeta creada: ',personaPath)
    os.makedirs(personaPath)

cap = cv2.VideoCapture('prueba1.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 1

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, +h), (0,255,0),2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (250,250), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personaPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1
    cv2.imshow('frame', frame)

    w = cv2.waitKey(1)
    if w== 7 or count >= 25:
        break

    #cap.release()
    #cv2.destroyAllWindows()
