import cv2
import os
import numpy as np


dataPath = 'C:/Users/iUser/Documents/GitHub/biometrico/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personaPath = dataPath + '/' + nameDir
    print('Descripción de imagen')

    for fileName in os.listdir(personaPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personaPath + '/' + fileName,0))
        image = cv2.imread(personaPath+ '/'+ fileName,0)
        cv2.imshow('image', image)
        cv2.waitKey(10)
        #cv2.destroyAllWindows
        #label = label + 1
        #print('labels=' , labels)
        #print('No. de etiquetas 0: ', np.count_nonzero(np.array(labels)==0))
        #print('No. de etiquetas 1: ', np.count_nonzero(np.array(labels)==1))

    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #entrenador
    print("Entrenando ...")
    face_recognizer.train(facesData, np.array(labels))

    #almacenando modelo
    face_recognizer.write('modeloEigenFace.xml')
    print("Modelo almacenando en proceso....")
