import mss
import numpy as np
import time
import winsound
import cv2
import pyautogui as pyg
pyg.PAUSE = 0

class GeoDashEnvironment:
    def __init__(self):
        self.framerate = 15
        self.lastframe = time.time()
        self.framewait = 1/15 - 0.005

        self.input_dims = (400,200)
        
        self.progress_bar_color = (3, 255, 127)
        self.progress_bar_pos = (184, 10)
        self.last_trigger = False
        
        self.reset()

    def get_inputs(self):
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1280, "height": 800}
            img_array = np.array(sct.grab(monitor))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            
        x, y = self.progress_bar_pos
        trigger_color = img_array[y][x]
        if (trigger_color == self.progress_bar_color).all():
            done = False
            self.last_trigger = True
        else:
            done = self.last_trigger
            self.last_trigger = False
        
        img_array = cv2.resize(img_array, dsize=self.input_dims)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array, done

    def reset(self):
        self.last_trigger = False
        observation, _ = self.get_inputs()
        return observation

    def step(self, action, display=False):
        if action == 1: pyg.press("space")
        time.sleep(max(0, self.framewait - (time.time()-self.lastframe) ))
        self.lastframe = time.time()
        observation, done = self.get_inputs()
        return observation, 0.5 + (-5*done), done

    def display(self):
        pass
        

if __name__ == "__main__":
    env = GeoDashEnvironment()
    while True:
        obs, reward, done = env.step(0)
        if done:
            winsound.Beep(440,500)
            env.reset()
