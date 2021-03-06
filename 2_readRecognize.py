import face_recognition
import cv2
import os
from os import walk
import biblioteca as biblio

#j = 0
#Encodings = []
#Names = []
#font = cv2.FONT_HERSHEY_SIMPLEX
image_dir = '/home/edgar/github/face_recognition/EDGAR/Unknown_faces/obama'
pickle_file = 'train.pkl'
#print(cv2.__version__)

biblio.compare_pickle_against_unknown(pickle_file, image_dir)
