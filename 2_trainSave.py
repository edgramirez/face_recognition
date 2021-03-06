import face_recognition
import os
from os import walk, path
import biblioteca as biblio 


Encodings = []
Names = []


def encode_known_faces(known_faces_path):
    files, root = biblio.read_images_in_dir(known_faces_path)

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

    if Names:
        print(Names)
        biblio.write_to_pickle(Names, Encodings)
    else:
        print('Ningun archivo de imagen contine rostros')



#encode_known_images("/home/edgar/github/face_recognition/EDGAR/known_faces/obama")
encode_known_faces("/home/edgar/github/face_recognition/EDGAR/known_faces/obamas_people")


