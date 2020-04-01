import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lecturaImg():
    os.chdir('train')
    for img in os.listdir('.'):
        imgRead = cv2.imread(img, 0)
        lecturaCanny = canny(imgRead)
        lecturaHarris = cornnerHarris(imgRead)
        #puntosInteres(lecturaCanny, imgRead)

        orb_ = orb(imgRead)


def orb(img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    plt.imshow(img2), plt.show()

def puntosInteres(img1, img2):
    detector = cv2.ORB_create()  # cambiar por otro como ORB (esquinas)

    # find the keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3), plt.show()

def canny(img):
    filtered_image = cv2.Canny(img, 100, 200)
    plt.imshow(filtered_image, cmap="gray")
    plt.show()

    return filtered_image

def cornnerHarris(img):
    blockSize = 3  # Tamaño de la ventana
    ksize = 3 # Tamaño del kernel de Sobel
    k = 0.05  # Factor de harris

    cornerness = cv2.cornerHarris(img,blockSize,ksize,k) #hace lo mismo que esta funcion

    threshold = 0.05
    indices = cornerness > threshold * cornerness.max()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    coords = [(j, i) for i in range(0, indices.shape[0]) for j in range(0, indices.shape[1]) if indices[i, j]]
    x = [j for j, i in coords]
    y = [i for j, i in coords]

    plt.plot(x, y, 'o')
    plt.show()

    return coords

def main():
    lecturaImg()


if __name__ == "__main__":
    main()

