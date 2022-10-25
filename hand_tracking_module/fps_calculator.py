import time

class FPS_CALCULATOR():
    def __init__(self,previousTime = 0, currentTime = 0):
        self.previousTime = previousTime
        self.currentTime = currentTime
    def calculate(self):
        self.currentTime = time.time()
        fps = 1/(self.currentTime - self.previousTime)
        self.previousTime = self.currentTime
        return fps