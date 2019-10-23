import cv2
import random
import time
from enum import IntEnum, auto
import numpy as np
from PIL import Image
import csv

from CvPutJaText import CvPutJaText

class State(IntEnum):
    START = auto()
    MAIN = auto()
    RESULT = auto()

class MainState(IntEnum):
    INIT = auto()
    COUNT = auto()
    GAME = auto()
    FINISH = auto()

X = 0
Y = 1
W = 0
H = 1

class GameState:
    def __init__(self):
        self.title = 'FACE GAME'

        self.frame_size = (640, 480)
        self.star_size = (70, 70)
        self.star_border = (100, 100)
        self.face_border = (50, 50)

        self.MAIN_STATE = MainState.INIT

        self.over_flag = True
        self.face_flag = False
        self.start_flag = True

        self.score = 0

        self.start_time = None
        self.left_time = 60

        self.start_limit = 3
        self.count_limit = 5
        self.game_limit = 60
        self.finish_limit = 3
        self.result_limit = 3
        self.limit_time = self.start_limit

        self.star_position = [0, 0]

        self.ranking = []
        self.ranking_file = './ranking.csv'

        self.result_change_state = 0

        self.frame = None
        self.ret = None

        self.face_list = None

        self.font_file = './OsakaMono.ttf'

        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, self.frame_size[X])
        self.capture.set(4, self.frame_size[Y])

        self.cascade_file = 'haarcascade_frontalface_alt2.xml'
        self.cascade = cv2.CascadeClassifier(self.cascade_file)

        self.img_star = cv2.imread('star.png', cv2.IMREAD_UNCHANGED)
        self.img_star = cv2.resize(self.img_star, self.star_size)

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()
    
    def start(self):
        start_btn_area = [(230, 350), (410, 400)]

        self.captureCamera()
        self.recognizeFace()

        cv2.rectangle(self.frame, (100, 50), (530, 140), (255,255,255), thickness = -1)
        self.frame = CvPutJaText.puttext(self.frame, self.title, (200, 70), self.font_file, 50, (0,0,0))

        cv2.rectangle(self.frame, start_btn_area[0], start_btn_area[1], (255,255,255), thickness = -1)
        self.frame = CvPutJaText.puttext(self.frame, u'はじめる', (265,360),self.font_file, 30, (0,0,0))

        for (x, y, w, h) in self.face_list:
            if self.over_area(start_btn_area[0], start_btn_area[1], [x + w / 2, y + h / 2]):
                break
        else:
            self.start_time = round(time.time())

        self.left_time = self.limit_time - (round(time.time()) - self.start_time)

        if self.left_time <= 0:
            return State.MAIN

        cv2.imshow(self.title, self.frame)
        return State.START

    def main(self):
        self.left_time = self.limit_time - (round(time.time()) - self.start_time)

        self.captureCamera()
        self.recognizeFace()

        if self.MAIN_STATE == MainState.INIT:
            self.start_time = round(time.time())
            self.limit_time = self.count_limit
            self.MAIN_STATE = MainState.COUNT

        elif self.MAIN_STATE == MainState.COUNT:
            bar_size = [(0, 140),(640, 340)] 
            bar_color = (255, 255, 255)
            font_size = 150
            font_color = (0, 0, 0)
            cv2.rectangle(self.frame, bar_size[0], bar_size[1], bar_color, thickness = -1)

            if self.left_time <= 0:
                self.start_time = round(time.time())
                self.limit_time = self.game_limit
                self.MAIN_STATE = MainState.GAME
            elif self.left_time == 0:
                self.frame = CvPutJaText.puttext(self.frame, u'START', (150,160),self.font_file, font_size, font_color)
            else:
                self.frame = CvPutJaText.puttext(self.frame, str(self.left_time), (290, 160), self.font_file, font_size, font_color)

        elif self.MAIN_STATE == MainState.GAME:

            if self.left_time <= 0:
                self.start_time = round(time.time())
                self.limit_time = self.finish_limit
                self.MAIN_STATE = MainState.FINISH
            else:
                if self.over_flag:
                    self.star_position[X] = random.randint(self.star_border[X], self.frame_size[X] - self.star_border[X])
                    self.star_position[Y] = random.randint(self.star_border[Y], self.frame_size[Y] - self.star_border[Y])
                    self.over_flag = False
                
                self.frame = self.overlay(self.frame, self.img_star, self.star_position[X], self.star_position[Y])

                for (x, y, w, h) in self.face_list:
                    self.face_flag = True

                    if abs(int(x + w / 2) - self.star_position[X]) <= self.face_border[X] and abs((y + h / 2) - self.star_position[Y]) <= self.face_border[Y]:
                        self.over_flag = True
                        self.score += 1
                        break
                    
                self.statusBar()
        
        elif self.MAIN_STATE == MainState.FINISH:
            bar_size = [(0, 140),(640, 340)] 
            bar_color = (255, 255, 255)
            font_size = 150
            font_color = (0, 0, 0)
            cv2.rectangle(self.frame, bar_size[0], bar_size[1], bar_color, thickness = -1)
            self.frame = CvPutJaText.puttext(self.frame, u'FINISH', (100,160),self.font_file, font_size, font_color)

            if self.left_time <= 0:
                self.ranking = []

                with open(self.ranking_file) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        for col in row:
                            self.ranking.append(int(col))
                self.ranking.append(self.score)
                self.ranking.sort(reverse = True)

                self.limit_time = self.result_limit

                return State.RESULT

        cv2.imshow(self.title, self.frame)

        return State.MAIN

    def result(self):

        continue_btn_area = [(100, 350), (280, 400)]
        return_btn_area = [(350, 350), (530, 400)]

        self.captureCamera()
        self.recognizeFace()

        cv2.rectangle(self.frame, (100, 50), (530, 330), (255,255,255), thickness = -1)
        cv2.rectangle(self.frame, (100, 350), (280, 400), (255,255,255), thickness = -1)
        cv2.rectangle(self.frame, (350, 350), (530, 400), (255,255,255), thickness = -1)

        self.frame = CvPutJaText.puttext(self.frame, u'ゲーム結果', (210,60),self.font_file, 40, (0,0,0))
        self.frame = CvPutJaText.puttext(self.frame, u'もう１度遊ぶ', (130,365),self.font_file, 20, (0,0,0))
        self.frame = CvPutJaText.puttext(self.frame, u'タイトルに戻る', (370,365),self.font_file, 20, (0,0,0))

        self.frame = CvPutJaText.puttext(self.frame, str(self.score) + u' ポイント', (240, 110), self.font_file, 25, (0,0,0))
        
        for i, ranking_score in enumerate(self.ranking):
            if i >= 5:
                break
            self.frame = CvPutJaText.puttext(self.frame, str(i + 1) + u'位 ' + str(ranking_score) + u' ポイント', (220,150 + i * 30),self.font_file, 25, (0,0,0))

        for (x, y, w, h) in self.face_list:
            if self.over_area(continue_btn_area[X], continue_btn_area[Y], [x + w / 2, y + h / 2]) and not(self.over_area(return_btn_area[X], return_btn_area[Y], [x + w / 2, y + h / 2])):
                self.result_change_state = 1
                break
            elif not(self.over_area(continue_btn_area[X], continue_btn_area[Y], [x + w / 2, y + h / 2])) and self.over_area(return_btn_area[X], return_btn_area[Y], [x + w / 2, y + h / 2]):
                self.result_change_state = 2
                break
        else:
            self.start_time = round(time.time())

        self.left_time = round(self.limit_time - (round(time.time()) - self.start_time))

        
        if self.left_time <= 0:
            print(time.time())
            with open(self.ranking_file, 'w') as f:
                writer = csv.writer(f) 
                writer.writerow(self.ranking[:5])
            if self.result_change_state == 1:
                self.limit_time = self.count_limit
                self.init_data()
                return State.MAIN
            elif self.result_change_state == 2:
                self.limit_time = self.start_limit
                self.init_data()
                return State.START

        cv2.imshow(self.title, self.frame)
        
        return State.RESULT

    def init_data(self):
        self.MAIN_STATE = MainState.INIT

        self.over_flag = True
        self.face_flag = False
        self.start_flag = True

        self.score = 0

        self.start_time = time.time()

        self.result_change_state = 0

    def over_area(self, p0, p1, pf):
        '''
        if (((p[X] - (x[0] - self.face_border[X])) <= ((x[1] + self.face_border[X]) - (x[0] - self.face_border[X])))
            and ((p[Y] - (y[0] - self.face_border[Y])) <= ((y[1] + self.face_border[Y]) - (y[0] - self.face_border[Y])))):
            return True
        '''
        '''
        print(0,((p[X] - (x[0])) <= ((x[1]) - (x[0]))))
        print(1,((p[Y] - (y[0])) <= ((y[1]) - (y[0]))))
        if (((p[X] - (x[0])) <= ((x[1]) - (x[0])))
            and ((p[Y] - (y[0])) <= ((y[1]) - (y[0])))):
            return True
        '''
        if (0 <= (pf[X] - p0[X]) <= (p1[X] - p0[X])) and (0 <= (pf[Y] - p0[Y]) <= (p1[Y] - p0[Y])):
            return True
        return False

    def statusBar(self):
        bar_size = (640, 50)
        bar_color = (255, 255, 255)
        font_size = 30
        font_color = (0, 0, 0)

        cv2.rectangle(self.frame, (0, 0), bar_size, bar_color, thickness = -1)
        self.frame = CvPutJaText.puttext(self.frame, u'SCORE ' + str(self.score), (10, 10), self.font_file, font_size, font_color)
        self.frame = CvPutJaText.puttext(self.frame, u'TIME ' + str(self.left_time), (500, 10), self.font_file, font_size, font_color)

        if self.face_flag:
            self.frame = CvPutJaText.puttext(self.frame, u'顔認識中', (250, 10), self.font_file, font_size, font_color)
            self.face_flag = False

    def captureCamera(self):
        self.ret, self.frame = self.capture.read()
        self.frame = cv2.flip(self.frame, 1)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2BGRA)

    def recognizeFace(self):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.face_list = self.cascade.detectMultiScale(frame_gray, minSize = (100, 100))

        face_rectangle_color = (0, 0, 255)
        pen_w = 3

        for (x, y, w, h) in self.face_list:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), face_rectangle_color, thickness = pen_w)

    def overlay(self, frame, image, x, y):

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
        