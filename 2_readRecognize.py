import face_recognition
import cv2
import os
from os import walk
import biblioteca as biblio
#print(cv2.__version__)

Encodings = []
Names = []
font = cv2.FONT_HERSHEY_SIMPLEX
image_dir = '/home/edgar/github/face_recognition/EDGAR/Unknown_faces/obama'
pickle_file = 'train.pkl'


def compare_pickle_against_unknown(pickle_file, image_dir):
    names_encodings = biblio.read_pickle(pickle_file)
    Names = names_encodings[0]
    Encodings = names_encodings[1]

    files, root = biblio.read_images_in_dir(image_dir)
    for file_name in files:
        testImagePath = os.path.join(root, file_name)
        testImage = face_recognition.load_image_file(testImagePath)
        testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

        # try to get the location of the face if there is one
        face_locations = face_recognition.face_locations(testImage)

        # if got a face, loads the image, else ignores it
        if face_locations:
            allEncodings = face_recognition.face_encodings(testImage, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, allEncodings):
                name = 'desconocido'
                matches = face_recognition.compare_faces(Encodings, face_encoding)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = Names[first_match_index]

                cv2.rectangle(testImage, (left, top),(right, bottom),(0, 0, 255), 2)
                cv2.putText(testImage, name, (left, top-6), font, .75, (180, 51, 225), 2)

            cv2.imshow('Imagen',testImage)
            cv2.moveWindow('Imagen', 0 ,0)

            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
        else:
            print("Image to search does not contains faces")
            print(testImagePath)

compare_pickle_against_unknown(pickle_file, image_dir)
