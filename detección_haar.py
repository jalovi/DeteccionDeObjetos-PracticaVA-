import os
import cv2

def lecturaImg():

    # iniciamos el clasificador de coche con los coches.xml
    car_cascade=cv2.CascadeClassifier('coches.xml')
    mat_cascade=cv2.CascadeClassifier('matriculas.xml')
    os.chdir('test')
    for img in os.listdir('.'):
        #Leer todas las imagenes de la carpeta test
        imgRead = cv2.imread(img)
        gray=cv2.cvtColor(imgRead,cv2.COLOR_BGR2GRAY)
        #Lanzar el detector coche
        car=car_cascade.detectMultiScale(gray,1.1,7,cv2.CASCADE_SCALE_IMAGE,(30,80))
        print ("Detect{0}car".format(len(car)))
        drawImage(car,gray,imgRead,mat_cascade)



def drawImage(car,gray,image,mat_cascade):
    if len(car)>0:
        for x,y,w,h in car:
            #Dibujar la posición del coche
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            #La posición de la matricula la busca por la posicion del coche
            car_gray = gray[y:y+h,x:x+w]
            car_color = image[y:y+h,x:x+w]
            #Lanzar el detector de las matriculas
            mat=mat_cascade.detectMultiScale(car_gray)
            for Mx,My,Mw,Mh in mat:
                #Dibujar la posición las matriculas
                cv2.rectangle(car_color,(Mx,My),(Mx+Mw,My+Mh),(255,0,0),2)
    cv2.imshow("Carimg",image)
    key = cv2.waitKey()
    #la ventana del imagen se cierra hasta que llega un ESC
    if key == 27:
        cv2.destroyAllWindows()

def main():
    lecturaImg()


if __name__ == "__main__":
    main()