import cv2
import mediapipe as mp
from  fps_calculator import FPS_CALCULATOR

class handDetector():
    def __init__(self,static_image_mode = False ,maxHands = 2, model_complexity=1, detectionCon = 0.5 ,trackCon = 0.5):

        self.static_image_mode = static_image_mode
        self.maxHands = maxHands
        self.model_complexity=model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        ##  File Object for hands  ##
        self.mpHands = mp.solutions.hands

        ##  Object to the Hand class  ##
        self.handObject = self.mpHands.Hands(
                                        self.static_image_mode,
                                        self.maxHands,
                                        self.model_complexity,
                                        self.detectionCon,
                                        self.trackCon)

        ##  File Object for drawing_utils  ##
        self.mpDraw = mp.solutions.drawing_utils

        ##  sets landmarks color and other specs##
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

        ##  sets connections color and other specs ##
        self.connections_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    
    def findHands(self, img, draw = True):

        ##  Convert the color of image to RGB  ##
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ##  Get the Land Marks  ##
        self.results = self.handObject.process(imgRGB)
        
        ##  If results !== None  means hands detected  ##
        if self.results.multi_hand_landmarks:

            ##  results.multi_hand_landmarks is a list of hands each hand in  is a list of landmarks that contains 21 landmarks  ##
            for hand in self.results.multi_hand_landmarks:

                ##  draw  ##
                if draw:
                    ##  draw landmarks and connect lines  ##
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS, self.landmark_drawing_spec, self.connections_drawing_spec)  
        
        return img

    def findPostion(self, img, handNo = 0, draw = True): 

        ##  landmark List  ##
        lmList = []

        ## setting image dimensions  ##
        height, width, channels = img.shape

        ##  If results !== None  means hands detected  ##
        if self.results.multi_hand_landmarks:

            ## select a hand from hands  ##
            myHand =self.results.multi_hand_landmarks[handNo]

            ##  id is the index and lm is landmark which is one of 21 landmarks of each hand  ##
            for id, lm in enumerate(myHand.landmark):

                ## convert ratios into pixels  ##
                xCordinate, yCordinate =  int(lm.x * width), int(lm.y * height)

                ## append landmark [id, xCordinate, yCordinate] to  the list  ##
                lmList.append([id, xCordinate, yCordinate])

        return lmList



def main():
    ##  Camera Handle  ##
    cap = cv2.VideoCapture(0)

    ##  Instantiating handDetector ##
    detector = handDetector() 

    ## Instantiating FPS_CALCULATOR  ##
    FPS = FPS_CALCULATOR()

    while True:

        ##  Read frames fron the camera  ##
        success, img = cap.read()

        ##  If  reading frame successfully  ##
        if success :         
            
            ##  finding landmarks  ##
            img = detector.findHands(img)

            ##  Find Position  ##
            lmList = detector.findPostion(img)

            if len(lmList) > 0:
                xCordinate= lmList[4][1]
                yCordinate = lmList[4][2]
                cv2.circle(img,(xCordinate, yCordinate), 10, (255, 0, 255), cv2.FILLED)

            ##  Calculate FPS  ##
            fps = FPS.calculate()

            ##  Put fps on the image  ##
            cv2.putText(img, str(int(fps)), (0,40), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

            ##   Display the result  ##
            cv2.imshow("image",img)

        ##  Break out of the loop when 'q' if pressed  ##
        if cv2.waitKey(1) == ord('q'):
            break 

    cap.release()


if __name__ == "__main__" :
    main()