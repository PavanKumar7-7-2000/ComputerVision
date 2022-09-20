from logging import captureWarnings
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

#######################################################################################

cap =  cv2.VideoCapture(0)

cap.set(3,640)

cap.set(4,480)

fpsReader = cvzone.FPS()

# imgBG = cv2.imread("42750.jpg")

#imgBG = cv2.resize(imgBG, (640,480), interpolation = cv2.INTER_AREA)

segmentor = SelfiSegmentation()

###################################################################################
images = os.listdir("images")
imglist = []
for imgPath in images:
    img = cv2.imread(f"images/{imgPath}")
    img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
    imglist.append(img)

imgIndex = 0

length =len(imglist)
##################################################################################

while True:
    success, img = cap.read() 

    if not success:
        break

    imgOut= segmentor.removeBG(img, imglist[imgIndex], threshold=0.95)

    ###############################################################################

    # imgStacked = cvzone.stackImages([img,imgOut],2,1)

    # _, imgStacked = fpsReader.update(imgStacked, color = (0,0,255))

    # cv2.imshow("Image", imgStacked)
    
    ####################################################################################

    cv2.imshow("Image", imgOut)

    key =cv2.waitKey(1)

    if  key == ord('f'):
        if imgIndex < length -1:
            imgIndex+=1

    elif  key == ord('b'):
        if imgIndex > 0:
            imgIndex-=1

    elif  key == ord('q'):
        break

   
###############################################################################

cap.release()