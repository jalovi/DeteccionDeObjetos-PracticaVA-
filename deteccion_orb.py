import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lecturaImg():
    kpYdes_ = dict()
    os.chdir('train')
    for img in os.listdir('.'):
        imgRead = cv2.imread(img, 0)
        #lecturaCanny = canny(imgRead)
        #lecturaHarris = cornnerHarris(imgRead)
        #puntosInteres(lecturaCanny, imgRead)

        kp_, des_ = orb(imgRead)
        #kpYdes_.setdefault(kp_, des_)


def orb(img):
    # Initiate ORB detector
    orb = cv2.ORB_create(100, nlevels=4, firstLevel=0, scaleFactor=1.3)
    # find the keypoints with ORB
    kp = orb.detect(img)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2), plt.show()

    # Para ello creamos un FlannBasedMatcher utilizando la distancia de Hamming
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des = np.uint8(des)
    # Luego almacenamos los descriptores. Esto se podría hacer según se calculan los descriptores
    for d in des:
        flann.add([d])

    #utilizamos el BFMatcher para fuerza bruta y disminuir los descriptores
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Ya podríamos, por ejemplo, buscar los k descriptores más parecidos al [[8,8,8]]
    results = flann.knnMatch(des, k=3)

    # Podemos mostrar el resultado obtenido por pantalla mediante:
    for r in results:
        for m in r:
            print("Res - dist:", m.distance ," img: ", m.imgIdx, " queryIdx: ", m.queryIdx, " trainIdx:", m.trainIdx)

    return kp, des


def main():
    lecturaImg()


if __name__ == "__main__":
    main()

