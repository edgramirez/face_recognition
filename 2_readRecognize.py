import biblioteca as biblio

image_dir = '/home/edgar/github/face_recognition/EDGAR/Unknown_faces/obama'
pickle_file = 'train.pkl'

biblio.compare_pickle_against_unknown(pickle_file, image_dir)
