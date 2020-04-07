import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def lecturaImg():
    kpsTrain_ = []
    dessTrain_ = []
    kpsTest_ = []
    dessTest_ = []

    tabla = np.zeros((250, 500))
    os.chdir('train2')
    for img in os.listdir('.'):
        imgRead = cv2.imread(img, 0)

        kpsTrain_, dessTrain_, flann = orb(imgRead, kpsTrain_, dessTrain_)

        #hacer una tabla para la votacion
        tabla = tabla + tablaVotacion(tabla, kpsTrain_)

    #prueba para comparar y hacer el vector de votacion
    os.chdir('../train2')
    print(os.getcwd())
    for img in os.listdir('.'):
        imgReadTest = cv2.imread(img, 0)

        kpsTest_, dessTest_, flannTest_ = orb(imgReadTest, kpsTest_, dessTest_)

        centro = compararMatcher(imgRead, imgReadTest, kpsTrain_, kpsTest_)

def tablaVotacion(tabla, kps):
    for kp in kps:
        y = kp[0]
        x = kp[1]
        y = int(y)
        x = int(x)

        tabla[x,y] += 1.0
    return tabla

def compararMatcher(img1, img2, kps1, kps2):
    for kp2 in kps2:
        for kp1 in kps1:
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            des1 = kp1[8]
            des2 = kp2[8]
            matches = bf.knnMatch(des1, des2, k=2)

            centro = vectors(kp1, kp2)
            # Apply ratio test
            good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
            #img3 = cv2.drawMatchesKnn(img2, kp2, img1, kps1, good, None, flags=2)
            #plt.imshow(img3), plt.show()

def orb(img,kps_,dess_):
    # inicializamos ORB
    orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
    # encontramos los keypoints con ORB
    kp = orb.detect(img)
    # sacamos los descriptores y los key point de la imagen y los guardamos en la lista
    kp, des = orb.compute(img, kp)

    #recorremos los key point con sus atributos y los guardamos en el array con los descriptores(tupla)
    kps_ = kps(kp,des,kps_,img)

    # Para ello creamos un FlannBasedMatcher utilizando la distancia de Hamming
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for d in des:
        flann.add([d])
        dess_.append(d)

    return kps_, dess_, flann

def kps(kp,des,kps_, img):
    # recorremos los key point con sus atributos y los guardamos en el array
    for i, key in enumerate(kp):
        vectorX=225-key.pt[0]
        vectorY=110-key.pt[1]
        vector = [vectorX,vectorY]
        if vectorY==0:
            anguloVec = 0
        else:
            anguloVec= np.arctan(vectorX / vectorY)
        modulo= np.math.sqrt(pow(vectorX, 2) + pow(vectorY, 2))
        vectorPolar=[modulo,anguloVec]
        x, y = key.pt
        centroImagen = calcularCentro(x, y, img)
        k = (x, y, vectorPolar, key.size, key.angle, key.response, key.octave, key.class_id, centroImagen, des[i])

        kps_.append(k)

    # dibujamos los keypoint en la imagen
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2), plt.show()

    return kps_

def calcularCentro(kpX, kpY, img):
    altura, anchura = img.shape
    centroX = anchura/2
    centroY = altura/2
    centroImg = (centroX, centroY)

    vectorX = centroX - kpX
    vectorY = centroY - kpY
    vector = (vectorX, vectorY)

    modulo = np.sqrt(np.power((centroX - kpX), 2) + np.power((centroY - kpY), 2))

    if (centroY - kpY) == 0:
        angulo = 0
    else:
        angulo = np.arctan((centroX - kpX) / (centroY - kpY))

    return (modulo, vector, angulo, centroImg)


def vectors(kp0,kpI):
    angulo=kp0[2][1]+kp0[4]-kpI[4]
    modulo=(kpI[3]/kp0[3])*kp0[2][0]
    puntoX=modulo*np.cos(angulo)
    puntoY=modulo*np.sin(angulo)
    puntoVotacion=[kpI[0]+puntoX,kpI[1]+puntoY]

    return puntoVotacion


def main():
    lecturaImg()


if __name__ == "__main__":
    main()