import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

labelsPath = os.path.join('yolo-coco', 'coco.names')
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.join('yolo-coco', "yolov3.weights")
configPath = os.path.join("yolo-coco", "yolov3.cfg")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

path_df = []
class_total_df = []
class_df = []
xmin_df = []
xmax_df = []
ymin_df = []
ymax_df = []
confidence_df = []
x_df = []
y_df = []
w_df = []
h_df = []
H_df = []
W_df = []

try:
    if not os.path.exists("output_txt"):
        os.makedirs("output_txt")
        print("directory is ready!")
    if not os.path.exists("image_check"):
        os.makedirs("image_check")
        print("directory is ready!")
except OSError:
    print ('Error: Creating directory. ')
    
    
path_folder = 'images/'


for folder in os.listdir(path_folder):
    for img in os.listdir(path_folder+folder):
        path = 'images/'+folder+'/'+img
        print(path_folder+folder+'/'+img)
        image = cv2.imread(path_folder+folder+'/'+img)
        
        try:
            img.shape
        except AttributeError:
            shutil.move(path,'image_check')
        
        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        file = open("output_txt/"+os.path.splitext(img)[0]+".txt", "w")

        size = W*H

        if size < 50000:
            shutil.move(path,'image_check')

        if len(idxs) == 0:
            shutil.move(path,'image_check')

        if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "bicycle" or \
                LABELS[classIDs[i]] == "motorbike" or LABELS[classIDs[i]] == "truck" or \
                LABELS[classIDs[i]] == "bus" or LABELS[classIDs[i]] == "person":
                    if classIDs[i] == 0:
                        classID = 5
                    if classIDs[i] == 1:
                        classID = 4
                    if classIDs[i] == 2:
                        classID = 0
                    if classIDs[i] == 3:
                        classID = 3
                    if classIDs[i] == 5:
                        classIDs = 1
                    if classIDs[i] == 7:
                        classID = 2

                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    xmin = x
                    xmax = x + w
                    ymin = y
                    ymax = y + h

                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    x_yolo,y_yolo,w_yolo,h_yolo = convert((W,H), b)

                    path_df += [path]
                    class_total_df += [len(idxs)]
                    class_df += [classID]
                    xmin_df += [xmin]
                    xmax_df += [xmax]
                    ymin_df += [ymin]
                    ymax_df += [ymax]
                    x_df += [x_yolo]
                    y_df += [y_yolo]
                    w_df += [w_yolo]
                    h_df += [h_yolo]
                    W_df += [W]
                    H_df += [H]
                    confidence_df += [confidences[i]]
                    data_txt =[]

                    data_txt.append(str(classID)+" "+str(x_yolo)+" "+str(y_yolo)+" "+str(w_yolo)+" "+str(h_yolo))

                    data = pd.DataFrame({'path': path_df, 'width': W_df, 'height': H_df ,'total_class': class_total_df, \
                                         'class_object' : class_df, 'xmin': xmin_df, 'xmax': xmax_df, 'ymin': ymin_df, 'ymax': ymax_df, \
                                        'x': x_df, 'y':y_df, 'w': w_df, 'h': h_df, 'confidence': confidence_df})
                    listToStr = ' '.join(map(str, data_txt))
                    file.write(listToStr+'\n')


        data.to_csv('data.csv',index=False)
        file.close()