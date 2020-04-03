import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lecturaImg():
    kps_ = []
    dess_ = []

    os.chdir('train')
    for img in os.listdir('.'):
        imgRead = cv2.imread(img, 0)

        flann_= orb(imgRead,kps_, dess_)





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

    # Luego almacenamos los descriptores.
    for i, d in enumerate(kps_):
        matches = flann.knnMatch(d[i], k=2)


    return kp, des

def kps(kp,des,kps_, img):
    # recorremos los key point con sus atributos y los guardamos en el array
    i = 0
    for key in kp:
        x = key.pt[0]
        y = key.pt[1]
        k = (x, y, key.pt, key.size, key.angle, key.response, key.octave, key.class_id, np.array(des[i]))
        i += 1
        kps_.append(k)

    # dibujamos los keypoint en la imagen
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2), plt.show()

    return kps_

def main():
    lecturaImg()


if __name__ == "__main__":
    main()

