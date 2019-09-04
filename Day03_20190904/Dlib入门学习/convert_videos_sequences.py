__author__ = "Luke Liu"
#encoding="utf-8"
import skvideo.io
import numpy as np
import  os
saved_file="C:/Users/asus/Desktop/sequences_image"
videogen = skvideo.io.vreader('C:/Users/asus/Desktop/bottle.mp4',num_frames=30)

i = 0
for frame in videogen:
    skvideo.io.vwrite(os.path.join(saved_file,"{}.png".format(i)), frame)
    i = i + 1
