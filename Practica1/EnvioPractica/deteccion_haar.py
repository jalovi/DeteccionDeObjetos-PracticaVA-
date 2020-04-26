import os
import cv2

def lecturaImg(path_name):
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta test
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
    gray=cv2.cvtColor(imgRead,0)
    #Lanzar el detector coche
    #car=car_cascade.detectMultiScale(gray,1.1,7,cv2.CASCADE_SCALE_IMAGE,(30,80))
    car=car_cascade.detectMultiScale(gray)
    print ("Detect{0}car".format(len(car)))
    if len(car)>0:
        for x,y,w,h in car:
            #Dibujar la posición del coche
            cv2.rectangle(imgRead,(x,y),(x+w,y+h),(0,255,0),2)
            #La posición de la matricula la busca por la posicion del coche
            car_gray = gray[y:y+h,x:x+w]
            car_color = imgRead[y:y+h,x:x+w]
            #Lanzar el detector de las matriculas
            mat=mat_cascade.detectMultiScale(car_gray)
            for Mx,My,Mw,Mh in mat:
                #Dibujar la posición las matriculas
                cv2.rectangle(car_color,(Mx,My),(Mx+Mw,My+Mh),(255,0,0),2)


def main():
    lecturaImg("test")


if __name__ == "__main__":
    main()