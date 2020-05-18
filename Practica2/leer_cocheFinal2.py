import os
import cv2
import argparse
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor


def lecturaImg(path_name, click, lda, gnb):
    for img in os.listdir(path_name):
        # Leer todas las imagenes de la carpeta test
        i = img.split('.')
        if (i[1] == 'jpg'):
            imgRead = cv2.imread(path_name + "/" + img)
            detectImage(imgRead, lda, gnb, click, i[0])
            # if click == True:
            key = cv2.waitKey()
            # la ventana del imagen se cierra hasta que llega un ESC
            if key == 27:
                cv2.destroyAllWindows()


def detectImage(imgRead, lda, gnb, visualizar, img):
    # iniciamos el clasificador de coche y de la matricula
    car_cascade = cv2.CascadeClassifier('haar/coches.xml')
    mat_cascade = cv2.CascadeClassifier('haar/matriculas.xml')

    gray = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #plt.imshow(gray), plt.show()

    filterB = cv2.bilateralFilter(gray, 8, 15, 10)
    #plt.imshow(filterB), plt.show()

    equ = cv2.equalizeHist(filterB)
    #plt.imshow(equ), plt.show()

    # Lanzar el detector coche
    car = car_cascade.detectMultiScale(gray)

    mat_sample = []
    if len(car) > 0:
        listaMat = []

        for x, y, w, h in car:
            # Dibujar la posición del coche

            cv2.rectangle(imgRead, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # hacemos el circulo del centro
            listaMat.append((x,y,w,h))
            centro, centroX, centroY = centroImagen(imgRead,listaMat)

            #plt.imshow(centro), plt.show()

        # Lanzar el detector de las matriculas
        mat = mat_cascade.detectMultiScale(gray)

        if len(mat)>0:
            equ = cv2.bilateralFilter(equ,5, 51, 51)
            #plt.imshow(equ), plt.show()

        imgCopy = imgRead.copy()

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #plt.imshow(binary), plt.show()

        cv2.drawContours(imgCopy, contours, -1, (0, 255, 0), 2)
        #plt.imshow(imgCopy), plt.show()

        if visualizar:
            cv2.imshow('Gray', gray)
            cv2.waitKey(0)
            cv2.imshow('Filter Binario', filterB)
            cv2.waitKey(0)
            cv2.imshow('Equalization', equ)
            cv2.waitKey(0)
            cv2.imshow('Binario', binary)
            cv2.waitKey(0)
            cv2.imshow('Contornos', imgCopy)
            cv2.waitKey(0)

        # identificamos las letras correctas
        lista=[]
        if (len(mat)) > 0:
            for Mx, My, Mw, Mh in mat:
                cv2.rectangle(imgRead, (Mx, My), (Mx + Mw, My + Mh), (255, 0, 0), 1)
                lista = detectorMat(contours, Mx, My, Mh, Mw, lista)
                if len(lista) < 7:
                    lista = encontrarMatricula(lista, imgRead, equ, visualizar)
        else:
            datax = []
            datay = []
            for Cx, Cy, Cw, Ch in car:
                datax, datay = detectorCar(contours, Cx, Cy, Cw, Ch, lista, datax, datay)
                #plt.imshow(imgRead), plt.show()
            if len(lista) >= 7:
                lista = Ransac(datax, datay, lista)
                centro, centroX, centroY = centroImagen(imgRead, listaMat)
            else:
                lista = []
                lista = listaMat
                lista = encontrarMatricula(lista, imgRead, equ, visualizar)
                centro, centroX, centroY = centroImagen(imgRead, lista)
                lista = detectarMatDificil(imgRead, lista)
        if(len(lista)>0):
            matriculaTexto = escribirCaracteres(lista, imgRead, binary, lda, gnb, visualizar)

    else:
        listaNueva = []
        listaNueva = encontrarMatricula(listaNueva, imgRead, equ)
        centro, centroX, centroY = centroImagen(imgRead,listaNueva)
        #listaNueva = detectarMatDificil(imgRead, listaNueva)
        if(len(listaNueva)>0):
            matriculaTexto = escribirCaracteres(listaNueva, imgRead, equ, lda, gnb, visualizar)
    if(len(lista)>0):
        saveImg(img, centroX, centroY, matriculaTexto,len(matriculaTexto))

def escribirCaracteres(lista, imgRead, binary, lda, gnb, visualizar):
    mat_sample = []
    lista.sort(key=takeFirst)
    for i in range(0, len(lista)):
        Dx, Dy, Dw, Dh = lista[i]
        # cambiamos el tamaño de la zona detectado
        caracter = cv2.resize(binary[Dy:Dy + Dh, Dx:Dx + Dw], (10, 10), interpolation=cv2.INTER_LINEAR)
        #plt.imshow(caracter), plt.show()

        # guardamos el array de pixeles en una matriz
        mat_sample.append(caracter)
        cv2.rectangle(imgRead, (Dx, Dy), (Dx + Dw, Dy + Dh), (0, 255, 0), 1)
        #plt.imshow(imgRead), plt.show()

    if (len(mat_sample) > 0):
        caracteres = convertirMat(mat_sample)
        caracteres = np.array(caracteres).astype('float32')
        cr_test = lda.transform(caracteres)
        predic = gnb.predict(cr_test)
        predic = NumberTochar(predic)
        dibujarCaracter(predic,lista,imgRead)
        if visualizar:
            cv2.imshow('Caracteres', imgRead)
            cv2.waitKey(0)

    return predic


def Ransac(datax, datay, lista):
    detectado = []
    x = np.array(datax).reshape((len(datax), 1))
    y = np.array(datay)
    ransac = RANSACRegressor()

    ransac.fit(x, y)
    inlier_mask = ransac.inlier_mask_
    for i in range(0, len(lista)):
        if inlier_mask[i]:
            detectado.append(lista[i])
    return detectado


def encontrarMatricula(lista, imgRead, equ, visualizar):
    img = imgRead.copy()

    filterB = cv2.bilateralFilter(equ, 11, 17, 17)
    #plt.imshow(filterB), plt.show()

    #__, imgThresh = cv2.threshold(filterB, 127, 255, 0)
    #plt.imshow(imgThresh), plt.show()

    ret, otsu = cv2.threshold(filterB, 127, 255, cv2.THRESH_OTSU)
    #plt.imshow(otsu), plt.show()

    bin = cv2.adaptiveThreshold(otsu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #plt.imshow(bin), plt.show()

    cnts, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        #print(area)
        if area > 6500 and area < 10000:
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
            #plt.imshow(img), plt.show()
            aspect_ratio = float(w) / h
            if aspect_ratio > 1 and aspect_ratio < 4:
                if visualizar:
                    cv2.imshow('contornos', img)
                    cv2.waitKey(0)
                cv2.rectangle(imgRead, (x, y), (x + w, y + h), (0, 0, 255), 1)
                lista.append((x, y, w, h))

    return lista


def detectarMatDificil(img, lista):
    imagen = img.copy()
    listaMat = []

    for i in range(0, len(lista)):
        Dx, Dy, Dw, Dh = lista[i]
        imagenMat = imagen[Dy:Dy + Dh, Dx:Dx + Dw]
        #plt.imshow(imagenMat), plt.show()

        gray = cv2.cvtColor(imagenMat, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray), plt.show()

        equ = cv2.bilateralFilter(gray, 5, 14, 4)
        # plt.imshow(equ), plt.show()

        binary = cv2.adaptiveThreshold(equ, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        #plt.imshow(binary), plt.show()

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

        cv2.drawContours(imagenMat, contours, -1, (0, 255, 0), 2)
        #plt.imshow(imagenMat), plt.show()
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area > 20:
                aspect = h/w
                if aspect >= 1 and aspect < 4.5 and Dy > 0:
                    if h > 30 and h < 50:
                        cv2.rectangle(imagen, (Dx+x, Dy+y), (Dx+x + w, Dy+y + h), (0, 0, 255), 1)
                        listaMat.append((Dx+x,Dy+y,w,h))
                        #plt.imshow(imagen), plt.show()

    return listaMat


def detectorMat(contours, MX, MY, MH, MW, lista):
    #Condicion para sacar la zona correcta de la matricula
    for i in range(0,len(contours)):
        area=cv2.contourArea(contours[i])
        if area>20:
            Dx, Dy, Dw, Dh = cv2.boundingRect(contours[i])
            if Dx >MX and Dy>MY and Dx<(MX+MW) and Dy<(MY+MH):
                aspect=Dh/Dw
                if aspect >= 1 and aspect<3 and Dy>0:
                    if Dw>=(MW/9) or Dh>=(MH/2) and len(lista)<8:
                        lista.append((Dx, Dy, Dw, Dh))
    return lista


def dibujarCaracter(predict, posicion, mat_color):
    for i in range(0, len(posicion)):
        Dx, Dy, Dw, Dh = posicion[i]
        cv2.putText(mat_color, predict[i], (int(Dx), int(Dy)+30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

def centroImagen(img, lista):
    x,y,w,h = lista[0]
    centrox = x + (((x + w) - x) / 2)
    centroy = y + (((y + h) - y) / 2)
    centro = img[y:y + h, x:x + y]

    cv2.circle(img, (int(centrox), int(centroy)), 5, (0, 0, 255), -1)

    return centro, centrox, centroy

def detectorCar(contours, Cx, Cy, Cw, Ch, lista, datax, datay):
    # Condicion para sacar la zona correcta de la matricula
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 30 and area < 300:
            Dx, Dy, Dw, Dh = cv2.boundingRect(contours[i])
            if Dx > Cx and Dy > Cy and Dx < (Cx + Cw) and Dy < (Cy + Ch):
                aspect = Dh / Dw
                if aspect >= 1 and aspect < 3.2 and Dy > 0:
                    if Dw >= (10) or Dh >= (15):
                        lista.append((Dx, Dy, Dw, Dh))
                        datax.append(len(lista))
                        datay.append(((Dx + Dw) / 2, (Dy + Dh) / 2))
    return datax, datay


def detectorCaracter(contours):
    lista = []
    # Condicion para sacar la zona correcta de la matricula
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 50:
            Dx, Dy, Dw, Dh = cv2.boundingRect(contours[i])
            aspect = Dh / Dw
            if aspect >= 1 and aspect < 3 and Dy > 0:
                if Dw >= (10) or Dh >= (10):
                    lista.append((Dx, Dy, Dw, Dh))
    return lista


def takeFirst(elem):
    return elem[0]


def CargarTraining():
    path_name = "training_ocr"
    EtiquetasE = []
    matrizC = []
    for img in os.listdir(path_name):
        # Leer todas las imagenes de la carpeta trainning
        i = img.split('.')
        if (i[1] == 'jpg'):
            # Coger el nombre del fichero para sacar la primera letra como la etiqueta
            Etiqueta = (i[0].split('_')[0])
            imgRead = cv2.imread(path_name + "/" + img)

            gray = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 9)

            binary = ((binary > 128) * 255).astype(np.uint8)

            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            lista = detectorCaracter(contours)
            for i in range(0, len(lista)):
                Dx, Dy, Dw, Dh = lista[i]
                cv2.rectangle(imgRead, (Dx, Dy), (Dx + Dw, Dy + Dh), (0, 255, 0), 1)
                # Cambiar el tamaño de la zona detectada del caracter
                caracter = cv2.resize(binary[Dy:Dy + Dh, Dx:Dx + Dw], (10, 10), interpolation=cv2.INTER_LINEAR)
                # guardar los pixeles en un matriz y las etiquetas en otras
                matrizC.append(caracter.reshape(100))
                EtiquetasE.append(charToNumber(Etiqueta))
    # Entrenar los clasificadores
    lda = LinearDiscriminantAnalysis(n_components=None)
    gnb = GaussianNB()
    matrix_data = np.array(matrizC).astype('float32')
    matrix_resp = np.array(EtiquetasE).astype('float32')
    lda = lda.fit(matrix_data, matrix_resp)
    CR = lda.transform(matrix_data)
    gnb = gnb.fit(CR, matrix_resp)
    return lda, gnb


def convertirMat(mat_sample):
    caracteres = []
    # Convertir en un matriz de 1*100
    for i in mat_sample:
        caracteres.append(i.reshape(100))
    return caracteres


def NumberTochar(lista):
    Caracteres = []
    dic = {10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
           21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W"
        , 33: "X", 34: "Y", 35: "Z", 0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
           36: "E"}
    for i in lista:
        Caracteres.append(dic.get(i))
    return Caracteres


def charToNumber(num):
    dic = {"A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17, "I": 18, "J": 19, "K": 20
        , "L": 21, "M": 22, "N": 23, "O": 24, "P": 25, "Q": 26, "R": 27, "S": 28, "T": 29, "U": 30, "V": 31, "W": 32,
           "X": 33
        , "Y": 34, "Z": 35, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "ESP": 36}
    return dic.get(num)


def saveImg(nombre, xCentro, yCentro, matricula, longitudMatricula):
    carpeta = "ImagenesTxt"
    rute = os.getcwd()
    if not os.path.isdir(carpeta):
        os.mkdir(carpeta)
    os.chdir(carpeta)
    texto = ""
    archivo = open(f"{nombre}.txt", "a")
    for i in matricula:
        if i == "ESP" or len(texto) > 7:
            texto = texto
        else:
            texto = texto + i
    contenido = nombre + ".jpg " + str(xCentro) + " " + str(yCentro) + " " + str(texto) + " " + str(longitudMatricula)

    archivo.write(contenido)
    archivo.write("\n")
    archivo.close()
    os.chdir(rute)


def main():
    rute = os.getcwd()
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument(
        "--path",
        default="testing_full_system", #testing_full_system
        help="path file to test",
    )
    parser.add_argument(
        "--click",
        default="True",  # cambiar a False cuando probemos en consola
        help="option that user check the bottom",
    )
    parser2 = argparse.ArgumentParser(description="ocr")
    parser2.add_argument(
        "--path",
        default="testing_ocr",
        help="path file to test",
    )
    parser2.add_argument(
        "--click",
        default="True",  # cambiar a False cuando probemos en consola
        help="option that user check the bottom",
    )
    lda, gnb = CargarTraining()
    arg = parser.parse_args()
    arg2 = parser2.parse_args()
    lecturaImg(arg.path, arg.click, lda, gnb)
    os.chdir(rute)
    lecturaImg(arg2.path, arg2.click, lda, gnb)


if __name__ == "__main__":
    main()