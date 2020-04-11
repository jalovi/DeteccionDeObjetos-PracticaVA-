import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ImgTest():
    kpsTrain_ = []

    # Para ello creamos un FlannBasedMatcher utilizando la distancia de Hamming
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    os.chdir('train')
    for img in os.listdir('.'):
        i = img.split('.')
        if(i[1] == 'jpg'):
            imgRead = cv2.imread(img, 0)

            kps_, des_ = orb(imgRead)
            flann.add([des_])
            kpsT_ = kps(kps_, des_, imgRead)

            kpsTrain_.append(kpsT_)

    return kpsTrain_, flann

def imgTrain(kpsTrain_, flann):

    #prueba para comparar y hacer el vector de votacion
    os.chdir('../test')
    for img in os.listdir('.'):
        i = img.split('.')
        if (i[1] == 'jpg'):
            imgReadTest = cv2.imread(img, 0)

            kps_, des_ = orb(imgReadTest)

            tabla = np.zeros((int(imgReadTest.shape[0] / 10), int(imgReadTest.shape[1] / 10)), dtype=np.uint8)
            knnSearch(tabla, kpsTrain_, kps_, des_, flann, imgReadTest, img)


def orb(img):
    # inicializamos ORB
    orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
    # encontramos los keypoints con ORB
    kp = orb.detect(img)
    # sacamos los descriptores y los key point de la imagen y los guardamos en la lista
    kp, des = orb.compute(img, kp)

    return kp, des

def kps(kp,des, img):
    # recorremos los key point con sus atributos y los guardamos en el array
    kps_ = []
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
            anguloVec = np.arctan((vectorY) / (vectorX))

        modulo = np.sqrt(np.power((centroX - x), 2) + np.power((centroY - y), 2))
        vectorPolar=[modulo,anguloVec]

        k = (key.class_id, x, y, vector, vectorPolar, key.size, key.angle, key.response, key.octave, des[i])

        kps_.append(k)

    # dibujamos los keypoint en la imagen
    #img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img2), plt.show()

    return kps_


def knnSearch(tabla, kpsTrain, kpsTest, desTest, flann, img, imgOrigin):
    matches = flann.knnMatch(desTest, k=5)

    points = []
    vVotacion = kpsTrain
    for match in matches:
        for i in match:
            points.append(i)
    pointsKpTrain = acomularKey(points, vVotacion)
    pointsKpTest = getKey(points, kpsTest)

    for i in range(len(pointsKpTest)):
        #k = (key.class_id,x, y, vector, vectorPolar, key.size, key.angle, key.response, key.octave,  des[i])
        kpTestSize = pointsKpTest[i].size
        kpTrainSize = pointsKpTrain[i][5]
        kpTestAngle = pointsKpTest[i].angle
        kpTrainDistance = pointsKpTrain[i][4][0]  # modulo
        kpTrainVectorAngle = pointsKpTrain[i][4][1]
        kpTrainAngle = pointsKpTrain[i][6]
        angulo = kpTrainVectorAngle
        scala = kpTrainSize/kpTestSize

        angulo = angulo + kpTrainAngle - kpTestAngle
        modulo = kpTrainDistance * scala

        anguloX = np.cos(angulo)
        anguloY = np.sin(angulo)
        vectorX = (anguloX * modulo)
        vectorY = (anguloY * modulo)

        kpx = int((pointsKpTest[i].pt[0] + vectorX)/10)
        kpy = int((pointsKpTest[i].pt[1] + vectorY)/10)

        if (kpx > 0 and kpy > 0) and (kpx < tabla.shape[1] and kpy < tabla.shape[0]):
            tabla[kpy, kpx] += 1

    #reagrupamos tabla
    tabla = cv2.resize(tabla, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    max_index = np.unravel_index(tabla.argmax(), tabla.shape)
    #position = (max_index[0], max_index[1])
    cv2.circle(img, (max_index[1], max_index[0]), 50, (0, 255, 255), thickness=2)
    nombre = imgOrigin.split('.')
    #saveImg(img, imgOrigin)
    cv2.imshow(nombre[0], img)
    cv2.waitKey(0)

def acomularKey(point,vector_votacion):
    lista_aco=[]
    for data in point:
        lista_aco.append(vector_votacion[data.imgIdx][data.trainIdx])
    return lista_aco

def getKey(point,key):
    lista_aco=[]
    for data in point:
        lista_aco.append(key[data.queryIdx])
    return lista_aco


def saveImg(img, nombre):
    rute = os.getcwd()
    if not os.path.exists('../resultados'):
        os.mkdir('../resultados')
    os.chdir('../resultados')
    cv2.imwrite(nombre, img)
    os.chdir(rute)



def main():
    ksTrain, flann = ImgTest()
    imgTrain(ksTrain, flann)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()