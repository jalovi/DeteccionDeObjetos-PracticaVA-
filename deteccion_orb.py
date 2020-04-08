import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def lecturaImg():
    kpsTrain_ = []
    dessTrain_ = []
    kpsTest_ = []
    dessTest_ = []
    test = True

    # Para ello creamos un FlannBasedMatcher utilizando la distancia de Hamming
    #tenemos que crear solo uno, no uno por imagen
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    os.chdir('train')
    for img in os.listdir('.'):
        i = img.split('.')
        if(i[1] == 'jpg'):
            imgRead = cv2.imread(img, 0)

            kpsTrain_, des_ = orb(imgRead, kpsTrain_)
            dessTrain_.append(des_)
            flann.add([np.uint8(des_)])

    #prueba para comparar y hacer el vector de votacion
    os.chdir('..')
    os.chdir('test')
    print(os.getcwd())
    for img in os.listdir('.'):
        i = img.split('.')
        if (i[1] == 'jpg'):
            imgReadTest = cv2.imread(img, 0)

            kpsTest_, des_ = orb(imgReadTest, kpsTest_)
            dessTest_.append(des_)

            tabla = np.zeros((imgReadTest.shape[0], imgReadTest.shape[1]))
            tablaVotacion(tabla, kpsTrain_, kpsTest_, des_, flann, imgReadTest, img)

            #centro = compararMatcher(imgRead, imgReadTest, kpsTrain_, kpsTest_)

def orb(img,kps_):
    # inicializamos ORB
    orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
    # encontramos los keypoints con ORB
    kp = orb.detect(img)
    # sacamos los descriptores y los key point de la imagen y los guardamos en la lista
    kp, des = orb.compute(img, kp)

    #recorremos los key point con sus atributos y los guardamos en el array con los descriptores(tupla)
    kps_ = kps(kp,des,kps_,img)

    return kps_, des

def kps(kp,des,kps_, img):
    # recorremos los key point con sus atributos y los guardamos en el array
    for i, key in enumerate(kp):
        x, y = key.pt
        centroX = img.shape[0] / 2
        centroY = img.shape[1] / 2
        vectorX = centroX - x
        vectorY = centroY - y
        vector = [vectorX, vectorY]
        print(centroY - vectorY)
        print(centroX - vectorX)
        if vectorY==0:
            anguloVec = 0
        else:
            anguloVec= np.arctan((centroY - vectorY) / (centroX - vectorX))
        print(anguloVec)
        modulo = np.sqrt(np.power((centroX - vectorX), 2) + np.power((centroY - vectorY), 2))
        vectorPolar=[modulo,anguloVec]

        k = (x, y, vector, vectorPolar, key.size, key.angle, key.response, key.octave, key.class_id, des[i])

        kps_.append(k)

    # dibujamos los keypoint en la imagen
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2), plt.show()

    return kps_


#tablaVotacion(tabla, kpsTrain_, kpsTest_, des_, flann, imgReadTest)
def tablaVotacion(tabla, kpsTrain, kpsTest, desTest, flann, img, imgOrigin):
    matches = flann.knnMatch(desTest, k=5)

    for match in matches:
        for i in match:
            #k = (x, y, vector, vectorPolar, key.size, key.angle, key.response, key.octave, key.class_id, des[i])
            kpTestSize =kpsTest[i.queryIdx][4]
            kpTrainSize = kpsTrain[i.trainIdx][4]
            kpTestAngle = kpsTest[i.queryIdx][5]
            kpTrainDistance = kpsTrain[i.trainIdx][3][0]#modulo
            kpTrainVectorAngle = kpsTrain[i.trainIdx][3][1]
            kpTrainAngle = kpsTrain[i.trainIdx][5]
            kpTestDistance = kpsTest[i.queryIdx][3][0]
            scala = kpTestSize / kpTrainSize

            angulo = kpTrainVectorAngle + kpTrainAngle - kpTestAngle
            modulo = (kpTestSize * kpTrainDistance) / kpTrainSize
            vectorX = modulo * np.cos(angulo)
            vectorY = modulo * np.sin(angulo)


            kpx = int(np.divide(kpsTest[i.queryIdx][0] + vectorX, 1))
            kpy = int(np.divide(kpsTest[i.queryIdx][1] + vectorY, 1))

            if (kpx > 0 and kpy > 0) and (kpx < tabla.shape[1] and kpy < tabla.shape[0]):
                tabla[kpy, kpx] += 1

    #reagrupamos tabla
    tabla = cv2.resize(tabla, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    max_index = np.unravel_index(tabla.argmax(), tabla.shape)
    position = (max_index[0], max_index[1])
    print(position)
    cv2.circle(img, position, np.uint8(img.shape[0] / 20), (0, 0, 255), thickness=2)
    nombre = imgOrigin.split('.')
    cv2.imshow(nombre[0], img)
    cv2.waitKey(0)


'''def vectors(kp0,kpI):
    angulo=kp0[2][1]+kp0[4]-kpI[4]
    modulo=(kpI[3]/kp0[3])*kp0[2][0]
    puntoX=modulo*np.cos(angulo)
    puntoY=modulo*np.sin(angulo)
    puntoVotacion=[kpI[0]+puntoX,kpI[1]+puntoY]
    return puntoVotacion'''


def main():
    lecturaImg()


if __name__ == "__main__":
    main()