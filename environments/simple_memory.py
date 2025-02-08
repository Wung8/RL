import numpy as np
import random

class SimpleMemoryEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.goal = random.choice([-1,1])
        self.pos = 0
        self.t = 0
        return self.get_inputs()

    def get_inputs(self):
        if self.t <= 2:
            if self.goal == -1:
                return (-1,0)
            else:
                return (1,0)
        return (0, self.pos)

    def step(self, actions, display=False):
        self.t += 1
        if self.t <= 4:
            actions = 1
            self.pos = 0
        else:
            if actions == 0:
                if self.pos >= 0:
                    self.pos -= 1
            if actions == 2:
                if self.pos <= 0:
                    self.pos += 1

        if display: self.display()

        r = self.pos * self.goal
        done = 1 if self.t >= 8 else 0
        return self.get_inputs(), r, done

    def display(self):
        toreturn = list('+.-')
        if self.goal == 1:
            toreturn = list('-.+')
        if self.t > 2:
            toreturn = list('...')
        if self.t > 4:
            toreturn[self.pos + 1] = 'x'
        else:
            toreturn[self.pos + 1] = 'o'
        print(' '.join(toreturn))
        
            
