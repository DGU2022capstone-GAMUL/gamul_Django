from django.shortcuts import render
from django.shortcuts import redirect, get_object_or_404
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse 
import cv2
import json
import numpy as np

# pweight = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/ver8_onion_radish/yolov3_custom_last (1).weights"
# pcfg = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/ver8_onion_radish/yolov3_custom.cfg"
# pcls = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/ver8_onion_radish/classes.names"
# pinput = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/testimage/testimage/test22.jpg" # test22만 객체 감지 가능 
# poutput = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/testimage/testimage/output.jpg"

@api_view(["GET", "POST"]) 
def product(request):
    
    
    # message = request.GET.get('')
    # image_url = '~~/.jpg'

    # res = requests.get(image_url)
    # test
    ## 고정
    pWeightpath = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/ver8_onion_radish/yolov3_custom_last (1).weights" 
    pCfgpath = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/ver8_onion_radish/yolov3_custom.cfg"
    pClasspath = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/ver8_onion_radish/classes.names" 
    ##
    
    ## 리액트에서 받을 것(jpg)
    pInputImage = "C:/Users/JYLEE/Desktop/gamul_jy/capstonefile/testimage/testimage/test22.jpg" # test22만 객체 감지 가능 
    
    
    
    # Load Yolo
    net = cv2.dnn.readNet(pWeightpath, pCfgpath) 
    classes = []
    with open(pClasspath, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    
    # Loading image
    img = cv2.imread(pInputImage)
    img = cv2.resize(img, None, fx=0.9, fy=0.9)
    height, width, channels = img.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers) 
    
    # 탐지한 객체의 클래스 예측 
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out: 
            scores = detection[5:]
            class_id = np.argmax(scores) # score중에 가장 큰 값을 가지는 index를 반환 
            confidence = scores[class_id]

            if confidence > 0.5: # names파일에서 원하는 객체 id에서 -1하기(선택사항)
                # 탐지한 객체 박싱
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
            
            
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    retval = {}
    retval['name'] = [] # 여러개 받을 수 있게
    retval['confidence'] = []
    
    for i in range(len(boxes)):
        if i in indexes:
            colors = np.random.uniform(0, 255, size=(len(classes), len(boxes)))
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = 255
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            cv2.putText(
                img, label + str(int(100*confidences[i])), (x, y + 30), font, 1, color, 3)
            
            retval["name"].append(label) # 백슬래쉬 제거가 안된다..
            retval["confidence"].append(int(100*confidences[i]))
            
    cv2.imshow("Image", img)


    cv2.waitKey(1) # cv2.waitkey(0) 오류 -> 해결 
    cv2.destroyAllWindows()

    if not confidences:
        print("농산물을 감지하지 못했습니다.") 
        retval.append("농산물 감지 실패")

    else:
        print(f"{label}을(를) 감지하였습니다.")


    # # result = JsonResponse({"result:result"})
    # result_serialized = serializers.serialize('json',result)
    # # return JsonResponse(result_serialized, safe=False)
    # return JsonResponse(json.loads(result))

    # serialed_result = resultSerializer(result, many=True)
    # return Response(data=serialed_result.data)
    result = json.dumps(retval)
    result = result.replace("\"", "") # \, "" 제거
    print(result)
    
    return Response(result)
    
    