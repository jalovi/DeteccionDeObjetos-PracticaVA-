import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def lecturaImg(path_name):
    vector_votacion=[]
    flann=createBase(vector_votacion)
    for img in os.listdir(path_name):
        #Leer todas las imagenes de la carpeta test
        imgRead = cv2.imread(path_name+"/"+img,0)
        detectImage(imgRead,flann,vector_votacion)
        cv2.imshow("Carimg",imgRead)
        key = cv2.waitKey()
        #la ventana del imagen se cierra hasta que llega un ESC
        if key == 27:
            cv2.destroyAllWindows()

def tabla_votacion(imgread,goodKeyTrain,goodKey):
    tamy=int((imgread.shape[0])/10)
    tamx=int((imgread.shape[1])/10)
    tabla = np.zeros((tamy,tamx))
    for index in range(len(goodKey)):
        scale=goodKeyTrain[index][4]/goodKey[index].size
        modulo,angulo=goodKeyTrain[index][1]
        angulo=angulo+goodKeyTrain[index][5]-goodKey[index].angle
        modulo=modulo*scale
        angX=np.cos(angulo)
        angY=np.sin(angulo)
        x0=angX*modulo
        y0=angY*modulo
        ejex=int((goodKey[index].pt[0]+x0)/10)
        ejey=int((goodKey[index].pt[1]+y0)/10)
        ejex2=int((goodKey[index].pt[0]-x0)/10)
        ejey2=int((goodKey[index].pt[1]-y0)/10)
        if (ejex > 0 and ejey > 0) and (ejex < tamx and ejey < tamy):
            tabla[ejey, ejex] += 1
        if (ejex2 > 0 and ejey2 > 0) and (ejex2 < tamx and ejey2 < tamy):
            tabla[ejey2, ejex2] += 1
    m=np.where(tabla==np.max(tabla))
    return m

def detectImage(imgRead,flann,vector_votacion):
    orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
    kp,des=orb.detectAndCompute(imgRead,None)
    matches = flann.knnMatch(des, k=5)
    good = []
    for matches in matches:
        for i in matches:
            good.append(i)
    goodKeyTrain=acomularKey(good,vector_votacion)
    goodKey=getKey(good,kp)
    index=tabla_votacion(imgRead,goodKeyTrain,goodKey)
    ejex=index[1]
    ejey=index[0]
    for i in range(len(ejex)):
        x=ejex[i]*10
        y=ejey[i]*10
        cv2.circle(imgRead,(x,y),50,(0,255,0),2)


def acomularKey(good,vector_votacion):
    lista_aco=[]
    for data in good:
        lista_aco.append(vector_votacion[data.imgIdx][data.trainIdx])
    return lista_aco

def getKey(good,key):
    lista_aco=[]
    for data in good:
        lista_aco.append(key[data.queryIdx])
    return lista_aco

def createBase(vector_votacion):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for img in os.listdir("train"):
        imgRead = cv2.imread("train/"+img, 0)
        orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
        kp,des=orb.detectAndCompute(imgRead,None)
        flann.add([des])
        vector=kps(kp,imgRead)
        vector_votacion.append(vector)
    return flann

def kps(kp,img):
    # recorremos los key point con sus atributos y los guardamos en el array
    kps =[]
    for i, key in enumerate(kp):
        x, y = key.pt
        centroX = img.shape[1] / 2
        centroY = img.shape[0] / 2
        vectorX = centroX - x
        vectorY = centroY - y
        vector = [vectorX, vectorY]
        if vectorX==0:
            anguloVec = 0
        else:
            anguloVec= np.arctan((vectorY) / (vectorX))
        #print(anguloVec)
        modulo = np.sqrt(np.power((centroX - x), 2) + np.power((centroY - y), 2))
        vectorPolar=[modulo,anguloVec]

        k = (key.class_id,vectorPolar,vector,key.pt,key.size, key.angle, key.response, key.octave)
        kps.append(k)
    return kps
def main():
    lecturaImg("test")




if __name__ == "__main__":
    main()


