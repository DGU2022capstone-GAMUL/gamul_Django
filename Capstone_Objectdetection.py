#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
# "C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\ver8_onion_radish\yolov3_custom_last (1).weights"
pweight = "C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\ver8_onion_radish\yolov3_custom_last (1).weights"
# "C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\ver8_onion_radish\yolov3_custom.cfg"
pcfg ="C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\ver8_onion_radish\yolov3_custom.cfg"
# "C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\ver8_onion_radish\classes.names"
pcls ="C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\ver8_onion_radish\classes.names"
pinput = "C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\testimage\testimage\test24.jpg"
poutput = "C:\Users\JYLEE\Desktop\gamul_Django\capstonefile\capstonefile\testimage\testimage\output.jpg"


def DetectObject(pWeightpath, pCfgpath, pClasspath, pInputImage, pOutputImage):

    # Load Yolo
    #"C:/Users/seyac/OneDrive - dongguk.edu/capstonefile/ver8_onion_radish/yolov3_custom_last (1).weights"
    #"C:/Users/seyac/OneDrive - dongguk.edu/capstonefile/ver8_onion_radish/yolov3_custom.cfg"
    #"C:/Users/seyac/OneDrive - dongguk.edu/capstonefile/ver8_onion_radish/classes.names"
    #"C:/Users/seyac/OneDrive - dongguk.edu/capstonefile/testimage/test24.jpg"
    #"C:/Users/seyac/OneDrive - dongguk.edu/capstonefile/testimage/output.jpg"
    net = cv2.dnn.readNet(pWeightpath, pCfgpath)
    classes = []
    with open(pClasspath, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    #output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Loading image
    img = cv2.imread(pInputImage)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # print(confidences)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    retval = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img, label + "/"+str(int(100*confidences[i])), (x, y + 30), font, 1, color, 3)
            retval.append(label + "/"+str(int(100*confidences[i])))
    cv2.imshow("Image", img)
    print(retval)
    cv2.imwrite(pOutputImage, img)  # 폴더에 output.png 이미지 저장
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not confidences:
        print("Error")
        retval.append("Error")
    else:
        print("no error")
    return retval


DetectObject(pweight, pcfg, pcls, pinput, poutput)


