import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lecturaImg():
    kpsTrain_ = []
    dessTrain_ = []
    kpsTest_ = []
    dessTest_ = []

    tabla = np.zeros((500, 500))
    os.chdir('train')
    for img in os.listdir('.'):
        imgRead = cv2.imread(img, 0)

        kpsTrain_, dessTrain_ = orb(imgRead, kpsTrain_, dessTrain_)

        #hacer una tabla para la votacion
        tabla= tabla + tablaVotacion(tabla, kpsTrain_)


    '''#prueba para comparar y hacer el vector de votacion
    os.chdir('../train2')
    print(os.getcwd())
    for img in os.listdir('.'):
        imgReadTest = cv2.imread(img, 0)

        kpsTest_, dessTest_ = orb(imgReadTest, kpsTest_, dessTest_)

        centro = compararMatcher(imgRead, imgReadTest, kpsTrain_, kpsTest_)'''

def tablaVotacion(tabla, kps):
    for kp in enumerate(kps):
        y = kp[1][0]
        x = kp[1][1]
        y = int(y)
        x = int(x)

        tabla[x,y] += 1.0
    return tabla

def compararMatcher(img1, img2, kps1, kps2):
    for i, kp2 in enumerate(kps2):
        for j, kp1 in enumerate(kps1):
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(kps1[8], kps2[8], k=2)

            # Apply ratio test
            good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
            img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good, None, flags=2)
            plt.imshow(img3), plt.show()

def orb(img,kps_,dess_):
    # inicializamos ORB
    orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
    # encontramos los keypoints con ORB
    kp = orb.detect(img)
    # sacamos los descriptores y los key point de la imagen y los guardamos en la lista
    kp, des = orb.compute(img, kp)

    #recorremos los key point con sus atributos y los guardamos en el array con los descriptores(tupla)
    kps_ = kps(kp,des,kps_,img)

    '''# Para ello creamos un FlannBasedMatcher utilizando la distancia de Hamming
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
    flann = cv2.FlannBasedMatcher(index_params, search_params)'''

    for d in des:
        #flann.add([d])
        dess_.append(d)

    return kps_, dess_

def kps(kp,des,kps_, img):
    # recorremos los key point con sus atributos y los guardamos en el array
    i = 0
    for key in kp:
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
        k = (x, y, vectorPolar, key.size, key.angle, key.response, key.octave, key.class_id, np.array(des[i]))

        kps_.append(k)
        i += 1

    # dibujamos los keypoint en la imagen
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2), plt.show()

    return kps_

def vectors(kp0,kpI):
    angulo=kp0.vectorPolar[1]+kp0.angle-kpI.angle
    modulo=(kpI/kp0)*kp0.vectorPolar[0]
    puntoX=modulo*np.cos(angulo)
    puntoY=modulo*np.sen(angulo)
    puntoVotacion=[kpI.x+puntoX,kpI.y+puntoY]

    return puntoVotacion

def main():
    lecturaImg()


if __name__ == "__main__":
    main()