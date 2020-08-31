import os
import cv2
import numpy as np
import random
import math
from dataset.align_faces import align_face


def read_file(file_path):
    path_images = []
    words = []
    lines = list(open(file_path, 'r').readlines())
    labels = []
    flag = False
    for line in lines:
        line = line.rstrip()
        if line.startswith('#') or line.startswith('/'):
            if flag == False:
                flag = True
            else:
                words.append(labels)
                labels = []
            image_name = line[2:]
            path_images.append(os.path.join(os.path.dirname(file_path), "images", image_name))
        else:
            label = [float(x) for x in line.split(' ')]
            labels.append(label)
    words.append(labels)
    return path_images, words


def pointInRect(point, rect):
    x1, y1, x2, y2, *t = rect
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def get_random_center(image, label):
    center = []
    w, h, _ = image.shape
    while len(center) != len(label):
        center_checked = False
        x, y = 0, 0
        while not center_checked:
            x = random.randrange(67, w - 67)
            y = random.randrange(67, h - 67)
            for box in label:
                if pointInRect((x, y), box):
                    continue
                if math.hypot(x - ((box[0] + box[2]) / 2), y - ((box[1] + box[3]) / 2)) < (h / 10):
                    continue
            center_checked = True
        center.append((x, y))
    return center


def _pad_to_square(image):
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = 0
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def generate_data(image, label):
    centers = get_random_center(image, label)
    face_images = []
    nonface_images = []
    for center in centers:
        try:
            crop_img = img[center[1] - 56: center[1] + 56, center[0] - 56: center[0] + 56]
            # crop_img = cv2.resize(crop_img, (32, 32))
            nonface_images.append(crop_img)
        except:
            print("FUCKING errror")

    faces = align_face(img, label, debug=False)
    # face = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
    # face = _pad_to_square(face)
    for face in faces:
        # face = cv2.resize(face, (32, 32))
        face_images.append(face)

    return face_images, nonface_images


if __name__ == "__main__":
    path_images, labels = read_file("/home/can/AI_Camera/EfficientFaceNet/data/widerface/train/label.txt")
    saved_train = '/home/can/AI_Camera/face_clasification/dataset/val'
    face_index = 0
    nonface_index = 15000
    for index, image_path in enumerate(path_images):
        print(image_path)
        img = cv2.imread(image_path)
        label = labels[index]

        annotations = np.zeros((0, 15))
        if len(label) == 0:
            continue
        for idx, box in enumerate(label):
            annotation = np.zeros((1, 15))
            # bbox
            if box[2] < 70 or box[3] < 70:
                continue
            annotation[0, 0] = int(box[0])  # x1
            annotation[0, 1] = int(box[1])  # y1
            annotation[0, 2] = int(box[0]) + int(box[2])  # x2
            annotation[0, 3] = int(box[1]) + int(box[3])  # y2

            # landmarks
            annotation[0, 4] = box[4]  # l0_x
            annotation[0, 5] = box[5]  # l0_y
            annotation[0, 6] = box[7]  # l1_x
            annotation[0, 7] = box[8]  # l1_y
            annotation[0, 8] = box[10]  # l2_x
            annotation[0, 9] = box[11]  # l2_y
            annotation[0, 10] = box[13]  # l3_x
            annotation[0, 11] = box[14]  # l3_y
            annotation[0, 12] = box[16]  # l4_x
            annotation[0, 13] = box[17]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        faces, non_faces = generate_data(img, target)
        for face in faces:
            image_path = os.path.join(saved_train, "face", "image_" + str(face_index) + ".jpg")
            # cv2.imwrite(image_path, face)
            face_index += 1
        for image in non_faces:
            image_path = os.path.join(saved_train, "nonface", "image_" + str(nonface_index) + ".jpg")
            try:
                cv2.imwrite(image_path, image)
                nonface_index += 1
            except:
                pass
