import os
import cv2
import detección_haar
import deteccion_orb

def lecturaVideo(path_name):
    cap=cv2.VideoCapture(path_name)
    while True:
        if cap.grab():
            flag,frame=cap.retrieve()
            if not flag:
                break
            else:
                detección_haar.detectImage(frame)
                cv2.imshow("CarVid",frame)
                key = cv2.waitKey()
                #la ventana del imagen se cierra hasta que llega un ESC
                if key == 27:
                    cv2.destroyAllWindows()


def lecturaVideo2(path_name):
    cap=cv2.VideoCapture(path_name)
    vector_votacion=[]
    vector_votacion, flann = deteccion_orb.ImgTest(vector_votacion)
    while True:
        if cap.grab():
            flag,frame=cap.retrieve()
            if not flag:
                break
            else:
                #knnSearch(kpsTrain, kpsTest, desTest, flann, img):
                kps, des = deteccion_orb.orb(frame)
                deteccion_orb.knnSearch(vector_votacion, kps, des, flann, frame)
                cv2.imshow("CarVid",frame)
                key = cv2.waitKey()
                #la ventana del imagen se cierra hasta que llega un ESC
                if key == 27:
                    cv2.destroyAllWindows()


def main():
    lecturaVideo("videos/video2.wmv")
    #lecturaVideo2("videos/video1.wmv")

if __name__ == "__main__":
    main()