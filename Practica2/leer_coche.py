import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lecturaImg(path_name):
    vector_votacion=[]

    #recorreos las imagenes de test
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta test
        imgRead = cv2.imread(path_name+"/"+img,0)
        #rectangulo matricula sacado de practica 1.2
        rectanguloMatricula(imgRead)
        #localizar todos los caracteres de la matricula
        localizarCaracteres(imgRead)
        cv2.imshow("Carimg",imgRead)

def rectanguloMatricula(img):
    car = car_cascade.detectMultiScale(img)
    print("Detect{0}car".format(len(car)))
    if len(car) > 0:
        for x, y, w, h in car:
            # Dibujar la posición del coche
            cv2.rectangle(imgRead, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # La posición de la matricula la busca por la posicion del coche
            car_gray = gray[y:y + h, x:x + w]
            car_color = imgRead[y:y + h, x:x + w]
            # Lanzar el detector de las matriculas
            mat = mat_cascade.detectMultiScale(car_gray)
            for Mx, My, Mw, Mh in mat:
                # Dibujar la posición las matriculas
                cv2.rectangle(car_color, (Mx, My), (Mx + Mw, My + Mh), (255, 0, 0), 2)

def localizarCaracteres(img):
    pass

def main():
    lecturaImg("testing_ocr")

if __name__ == "__main__":
    main()


