import os
import cv2
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lecturaImg(path_name, click):
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta test
        i = img.split('.')
        if(i[1] == 'jpg'):
            imgRead = cv2.imread(path_name+"/"+img)
            detectImage(imgRead, img)
            cv2.imshow("Carimg",imgRead)
            if click == True:
                key = cv2.waitKey()
                #la ventana del imagen se cierra hasta que llega un ESC
                if key == 27:
                    cv2.destroyAllWindows()


def detectImage(imgRead, img):
    # iniciamos el clasificador de coche y de la matricula
    car_cascade=cv2.CascadeClassifier('haar/coches.xml')
    mat_cascade=cv2.CascadeClassifier('haar/matriculas.xml')
    gray=cv2.cvtColor(imgRead,cv2.COLOR_BGR2GRAY)
    #Lanzar el detector coche
    #car=car_cascade.detectMultiScale(gray,1.1,7,cv2.CASCADE_SCALE_IMAGE,(30,80))
    car=car_cascade.detectMultiScale(gray)
    print ("Detect{0}car".format(len(car)))
    if len(car)>0:
        for x,y,w,h in car:
            #Dibujar la posición del coche
            cv2.rectangle(imgRead,(x,y),(x+w,y+h),(0,0,255),2)
            #Lanzar el detector de las matriculas
            mat=mat_cascade.detectMultiScale(gray)
            for Mx,My,Mw,Mh in mat:
                #Dibujar la posición las matriculas
                cv2.rectangle(imgRead,(Mx,My),(Mx+Mw,My+Mh),(255,0,0),2)
                mat_gray=gray[My:My+Mh,Mx:Mx+Mw]
                mat_color = imgRead[My:My+Mh,Mx:Mx+Mw]
                binary = cv2.adaptiveThreshold(mat_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                lista=detectorCaracter(contours, img)
                for i in range(0,len(lista)):
                    Dx, Dy, Dw, Dh = cv2.boundingRect(lista[i])
                    cv2.rectangle(mat_color, (Dx,Dy), (Dx+Dw,Dy+Dh), (0,255,0), 2)

def detectorCaracter(contours, nombreImagen):
    lista=[]
    for i in range(0,len(contours)):
        area=cv2.contourArea(contours[i])
        if area>50:
            Dx, Dy, Dw, Dh = cv2.boundingRect(contours[i])
            aspect=Dh/Dw
            if aspect > 0 and aspect<5.5 and Dy!=0:
                lista.append(contours[i])
                #saveImg(nombreImagen, Dx, Dy, contours[i], Dh)
    return lista

def CargarTraining():
    path_name="training_ocr"
    EtiquetasE=[]
    matrizC=[]
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta trainning
        i = img.split('.')
        if(i[1] == 'jpg'):
            caracter=i[0].split('_')
            EtiquetasE.append(caracter[0])
            imgRead = cv2.imread(path_name+"/"+img)
            imgRead=cv2.resize(imgRead,(10,10),interpolation=cv2.INTER_LINEAR)
            gray=cv2.cvtColor(imgRead,cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            lista=convertirMat(binary)
            matrizC.append(lista)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(matrizC,EtiquetasE)
    CR=lda.transform(matrizC)
    return matrizC
            #contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #lista=ClasificarCaracter(contours)
            #for i in range(0,len(lista)):
             #   Dx, Dy, Dw, Dh = cv2.boundingRect(lista[i])
              #  cv2.rectangle(imgRead, (Dx,Dy), (Dx+Dw,Dy+Dh), (0,255,0), 2)
            #cv2.imshow("Caracterimg",imgRead)
            #key = cv2.waitKey()
            #la ventana del imagen se cierra hasta que llega un ESC
            #if key == 27:
             #   cv2.destroyAllWindows()
def convertirMat(binarys):
    lista=[]
    for binary in binarys:
        for i in range(len(binary)):
            lista.append(binary[i])
    return lista



def ClasificarCaracter(contours):
    lista=[]
    for i in range(0,len(contours)):
        area=cv2.contourArea(contours[i])
        if area>20:
            lista.append(contours[i])
    return lista
    
def saveImg(nombre, xCentro, yCentro, matricula, longitudMatricula):
    rute = os.getcwd()
    if not os.path.exists('resultados'):
        os.mkdir('resultados')
    os.chdir('resultados')
    archivo = open(f"{nombre}.txt", "w")
    contenido = "Nombre_Imagen: "+nombre +", x_centro_matricula: " +str(xCentro) +", y_centro_matricula: " +str(yCentro) +", Matricula: " +str(matricula) +", longitud_Matricula: " +str(longitudMatricula)
    archivo.write(contenido)
    archivo.close()
    os.chdir(rute)


def main():
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument(
        "--path",
        default="testing_ocr",
        help="path file to test",
    )
    parser.add_argument(
        "--click",
        default="True",#cambiar a False cuando probemos en consola
        help="option that user check the bottom",
    )

    arg = parser.parse_args()
    lecturaImg(arg.path, arg.click)

    #lecturaImg("testing_ocr")
    CargarTraining()

if __name__ == "__main__":
    main()
