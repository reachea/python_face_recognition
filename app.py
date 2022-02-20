import face_recognition
import numpy as np
import cv2

known_img = face_recognition.load_image_file('KNOWN IMG 1')
known_img_encode = face_recognition.face_encodings(known_img)[0]

known_img_1 = face_recognition.load_image_file('KNOWN IMG 2')
known_img_1_encode = face_recognition.face_encodings(known_img_1)[0]

known_faces = [
  known_img_encode,
  known_img_1_encode,
]
known_names = [
  'Somon',
  'Reach',
]

if __name__ == "__main__":
  unknown_image = face_recognition.load_image_file('UNKNOWN IMG')
  name = "Unknown"

  face_locations = face_recognition.face_locations(unknown_image)

  if (face_locations):
    unknown_image_encode = face_recognition.face_encodings(unknown_image, face_locations)[0]

    matches = face_recognition.compare_faces(known_faces, unknown_image_encode)

    face_distance = face_recognition.face_distance(known_faces, unknown_image_encode)
    best_match_index = np.argmin(face_distance)
    if (matches[best_match_index]):
      name = known_names[best_match_index]

  print(name)
