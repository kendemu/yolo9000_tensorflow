from darkflow.net.build import TFNet
import cv2

yolo9000 = {"model" : "cfg/yolo9000.cfg", "load" : "weights/yolo9000.weights", "threshold": 0.01}
tiny_yolo = {"model" : "cfg/tiny-yolo-voc.cfg", "load" : "weights/tiny-yolo-voc.weights", "threshold": 0.2}

tfnet = TFNet(yolo9000)



img = "/home/demulab/cv_ws/before-bin.jpg"
imgcv = cv2.imread(img)
result = tfnet.return_predict(imgcv)
for res in result:
    if res["label"] == "whole":
        print "detected whole"
        continue
    else:
        color = int(255 * res["confidence"])
        top = (res["topleft"]["x"], res["topleft"]["y"])
        bottom = (res["bottomright"]["x"], res["bottomright"]["y"])
        cv2.rectangle(imgcv, top, bottom, (255-color, 0, color) , 2)
        cv2.putText(imgcv, res["label"], top, cv2.FONT_HERSHEY_DUPLEX, 1.0, (255-color, 0, color))

print result
cv2.imshow("image", imgcv)
cv2.waitKey(0)
cv2.destroyAllWindows()

