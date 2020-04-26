import os
import cv2

def lecturaImg(path_name):
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta test
        i = img.split('.')
        if(i[1] == 'jpg'):
            imgRead = cv2.imread(path_name+"/"+img)
            detectImage(imgRead)
            cv2.imshow("Carimg",imgRead)
            key = cv2.waitKey()
        #la ventana del imagen se cierra hasta que llega un ESC
            if key == 27:
                cv2.destroyAllWindows()


def detectImage(imgRead):
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
            #La posición de la matricula la busca por la posicion del coche
            car_gray = gray[y:y+h,x:x+w]
            car_color = imgRead[y:y+h,x:x+w]
            #Lanzar el detector de las matriculas
            mat=mat_cascade.detectMultiScale(car_gray)
            for Mx,My,Mw,Mh in mat:
                #Dibujar la posición las matriculas
                cv2.rectangle(car_color,(Mx,My),(Mx+Mw,My+Mh),(255,0,0),2)
                mat_gray=car_gray[My:My+Mh,Mx:Mx+Mw]
                mat_color = car_color[My:My+Mh,Mx:Mx+Mw]
                binary = cv2.adaptiveThreshold(mat_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                lista=detectorCaracter(contours)
                for i in range(0,len(lista)):
                    Dx, Dy, Dw, Dh = cv2.boundingRect(lista[i])
                    cv2.rectangle(mat_color, (Dx,Dy), (Dx+Dw,Dy+Dh), (0,255,0), 2)

def detectorCaracter(contours):
    lista=[]
    for i in range(0,len(contours)):
        area=cv2.contourArea(contours[i])
        if area>50:
            Dx, Dy, Dw, Dh = cv2.boundingRect(contours[i])
            aspect=Dh/Dw
            if aspect > 0 and aspect<5.5 and Dy!=0:
                lista.append(contours[i])
    return lista
def CargarTraining():
    path_name="training_ocr"
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta trainning
        i = img.split('.')
        if(i[1] == 'jpg'):
            imgRead = cv2.imread(path_name+"/"+img)
            detectImage(imgRead)
            cv2.imshow("Carimg",imgRead)
            key = cv2.waitKey()
            #la ventana del imagen se cierra hasta que llega un ESC
            if key == 27:
                cv2.destroyAllWindows()
def main():
    lecturaImg("testing_ocr")


if __name__ == "__main__":
    main()