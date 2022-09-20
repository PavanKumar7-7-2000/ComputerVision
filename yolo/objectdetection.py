import cv2
import numpy as np


cap =  cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

width = 320
height = 320


confThreshold=0.5
nmsThreshold = 0.3

classFile = "coco.names"
classes = []

with open(classFile,'rt') as file:
    classes = file.read().rstrip('\n').split('\n')

# print(classes)

# print(len(classes))

modelConfiguration  = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"


net = cv2.dnn.readNetFromDarknet(modelConfiguration ,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img,confThreshold=0.5,nmsThreshold = 0.3):
    hT,wT,cT = img.shape
    bbox =[]
    classIds = []
    confs = [] 
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    for i in indices:
        box=bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f"{classes[classIds[i]].upper()} {int(confs[i]*100)}%",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)



     

while True:
    success, img = cap.read() 

    if not success: 
        break

    blob = cv2.dnn.blobFromImage(img, 1/255, (width,height), [0,0,0], 1, crop = False)
    net.setInput(blob)

    layerNames = net.getLayerNames() 
    # print(net.getUnconnectedOutLayers())
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    

    findObjects(outputs,img)


    cv2.imshow("Image", img)


    key =cv2.waitKey(1)

    if  key == ord('q'):
        break


cap.release()
