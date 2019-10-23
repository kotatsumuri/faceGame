import cv2
import numpy as np
from PIL import Image
import random
import time
from CvPutJaText import CvPutJaText

def overlay(frame,image,x,y):

    layer1 = cv2.cvtColor(frame,cv2.COLOR_BGRA2RGB)
    layer2 = cv2.cvtColor(image,cv2.COLOR_BGRA2RGBA)

    layer1 = Image.fromarray(layer1)
    layer2 = Image.fromarray(layer2)

    layer1 = layer1.convert("RGBA")
    layer2 = layer2.convert("RGBA")

    tmp = Image.new("RGBA",layer1.size,(255,255,255,0))
    tmp.paste(layer2,(x,y))

    result = Image.alpha_composite(layer1,tmp)

    return cv2.cvtColor(np.asarray(result),cv2.COLOR_RGBA2BGRA)


capture = cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)

cascade_file = "haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_file)

img_principal = cv2.imread("principal.png", cv2.IMREAD_UNCHANGED)
img_star = cv2.imread("star.png", cv2.IMREAD_UNCHANGED)
img_star = cv2.resize(img_star, (70, 70))

over_flag = True
face_flag = False

star_x = 0
star_y = 0
point = 0

start_time = round(time.time())
limit_time = 60

while True:

    ret,frame = capture.read()
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(frame_gray,minSize = (100,100))
    
    if over_flag:
        star_x = random.randint(100, 540)
        star_y = random.randint(100, 380)
        over_flag = False

    frame = overlay(frame, img_star, star_x, star_y)
    
    for (x,y,w,h) in face_list:
        color = (0,0,225)
        pen_w = 3
        face_flag = True
        #img_pr = cv2.resize(img_principal,(w,h+50))
        #frame = overlay(frame,img_pr,x,y-50)
        
        cv2.rectangle(frame,(x,y),(x + w, y + h),color,thickness = pen_w)
        #cv2.rectangle(frame,(int(x + w / 2 - 50),int(y + h / 2 - 50)),(int(x + w / 2 + 50), int(y + h / 2 + 50)),color,thickness = -1)
        if x + w / 2 >= star_x - 50 and x + w / 2 <= star_x + 50 and y + h / 2 >= star_y - 50 and y + h / 2 <= star_y + 50:
            over_flag = True
            point += 1

    cv2.rectangle(frame,(0,0),(640, 50),(255,255,255),thickness = -1)
    frame = CvPutJaText.puttext(frame, u"SCORE " + str(point), (10, 10), './OsakaMono.ttf', 30, (0,0,0))
    frame = CvPutJaText.puttext(frame, u"TIME " + str(limit_time - (round(time.time()) - start_time)), (500, 10), './OsakaMono.ttf', 30, (0,0,0))
    
    if face_flag:
        frame = CvPutJaText.puttext(frame, u"顔認識中", (250, 10), './OsakaMono.ttf', 30, (0, 0, 0))
        face_flag = False
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
