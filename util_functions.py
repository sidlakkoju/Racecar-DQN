import cv2
import numpy as np
from collections import deque

def rgb_to_gray(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def get_frame_stack(frame_queue):
    frame_stack = np.array(frame_queue)
    return frame_stack
    
