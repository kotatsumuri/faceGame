from enum import IntEnum, auto
import cv2

from GameState import GameState, State

class GameMaster:
    def __init__(self):
        self.STATE = State.START
        self.gameState = GameState()
        self.stateControler()

    def stateControler(self):
        while True:
            if self.STATE == State.START:
                self.STATE = self.gameState.start()
            elif self.STATE == State.MAIN:
                self.STATE = self.gameState.main()
            elif self.STATE == State.RESULT:
                self.STATE = self.gameState.result()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
