import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier


from tkinter import *
root = Tk()

root.geometry("700x450")
root["bg"]="#1cedc3"
# maxsize(width,height
root.maxsize(700,850)
# changing title from tk to wanted string
root.title("Hand Gesture Recognition App")
#minsix=ze
root.minsize(250,150)



# label for print the text
label1 = Label(text="Hand Gesture Recognition")
label1.pack()


# buttons for camera open and close
open_camera_btn= Button(root, text="OPEN", command=open_camera,width=10,height=2,fg="white",bg="black",activebackground="blue")
close_camera_btn= Button(root, text="Press 'e' ", command=close_camera,width=10,height=2,fg="white",bg="black",activebackground="blue")
# open_camera_btn.place(x=300, y=300)
# close_camera_btn.place(x=300,y=300)
open_camera_btn.pack()
close_camera_btn.pack()
def open_camera():
    pass
def close_camera():
    pass

root.mainloop()
