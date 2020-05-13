# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('train')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("train/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("train/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# normalize input vectors
in_encoder = Normalizer(norm='l2')
encodings = in_encoder.transform(encodings)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(names)
names = out_encoder.transform(names)

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale', probability=True)
#clf = svm.SVC(kernel='linear', probability=True)
clf.fit(encodings,names)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('test/test.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    test_image_enc = in_encoder.transform([test_image_enc])
    name = clf.predict(test_image_enc)
    prob = clf.predict_proba(test_image_enc)
    print(prob)
    acc = prob[0,name[0]]
    name = out_encoder.inverse_transform(name)
    print(*name, acc)