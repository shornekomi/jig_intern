import numpy as np
import cv2 ,os, time
from PIL import Image

train_path = "./training/"
test_path = "./test_pic/"

cascadePath = "harcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

fisher_recognizer = cv2.createFisherFaceRecognizer()
recognizer = cv2.createLBPHFaceRecognizer()

import time

def get_images_and_labels(path):
    images = []
    labels = []
    files = []
    for f in os.listdir(path):
        image_path = os.path.join(path, f)
        print(image_path)
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
            images.append(roi)
            labels.append(int(f[0:1]))
            files.append(f)

    return images, labels, files

images, labels, files = get_images_and_labels(train_path)
recognizer.train(images, np.array(labels))
fisher_recognizer.train(images, np.array(labels))
recognizer.save("./face_LBPH.yml")
fisher_recognizer.save("./face_Fisher.yml")
print("\n to go test images")
test_images, test_labels, test_files = get_images_and_labels(test_path)

i = 0
cv2.namedWindow("test_image")

while i < len(test_images):
    if test_labels is None:
        break

    label, confidence = recognizer.predict(test_images[i])
    label_fish,confidence_fish = fisher_recognizer.predict(test_images[i])
    print("Test Image: {}, Predict Label: {}, Confidence: {}". format(test_files[i], label, confidence))
    print("TEst Image: {}, Predict Label: {}, COnfidence: {}". format(test_files[i], label_fish, confidence_fish))
    #cv2.imshow("test_image", test_images[i])
    print("test{}".format(i))
    key = cv2.waitKey(0)
    if key == 112:
        break    
print("test end")
