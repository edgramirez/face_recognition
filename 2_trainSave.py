import face_recognition
import cv2
import os
import pickle
from os import walk, path


#print(cv2.__version__)
Encodings = []
Names = []


def read_images_in_dir(path_to_read):
    dir_name, subdir_name, file_names = next(walk(path_to_read))
    images = [item for item in file_names if '.jpeg' in item[-5:] or '.jpg' in item[-4:] or 'png' in item[-4:] ]
    return images, dir_name

def encode_known_faces(known_faces_path):
    files, root = read_images_in_dir(known_faces_path)

    for file_name in files:
        path = root + '/' + file_name
        name = os.path.splitext(file_name)[0]

        # load the image into face_recognition library
        person = face_recognition.load_image_file(path)

        # try to get the location of the face if there is one 
        face_locations = face_recognition.face_locations(person)

        # if got a face, loads the image, else ignores it
        if face_locations:
            encoding = face_recognition.face_encodings(person)[0]
            Encodings.append(encoding)
            Names.append(name)
    print(Names)

    if Names:
        write_to_pickle(Names, Encodings)
    else:
        print('Ningun archivo de imagen contine rostros')

def write_to_pickle(Names, Encodings):
    with open('train.pkl','wb') as f:
        pickle.dump(Names, f)
        pickle.dump(Encodings, f)


#encode_known_images("/home/edgar/github/face_recognition/EDGAR/known_faces/obama")
encode_known_faces("/home/edgar/github/face_recognition/EDGAR/known_faces/obamas_people")
