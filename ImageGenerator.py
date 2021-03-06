import datetime
import os
import random
import time

import cv2
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import skimage as sk

# Brightness
BRIGHTNESS_PERCENT_LOWER = 0.4
BRIGHTNESS_PERCENT_HIGHER = 1.2

# Blur
KERNEL_SIZE_POSSIBILITIES = [7, 13, 15]
SIGMA_X_LOWER = 0
SIGMA_X_UPPER = 5
SIGMA_Y_LOWER = 0
SIGMA_Y_UPPER = 5

# Rotation
ANGLE_CHANGE_MAX = 90

# Translation
TRANSLATION_X_LOWER = 2.5
TRANSLATION_X_UPPER = 5.0
TRANSLATION_Y_LOWER = 2.5
TRANSLATION_Y_UPPER = 5.0

# Shearing
SHEAR_RANGE = 7


def change_brightness(image):
    percent_change = random.uniform(BRIGHTNESS_PERCENT_LOWER, BRIGHTNESS_PERCENT_HIGHER)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = (hsv[:, :, 2] * percent_change).astype(int)

    output_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return output_image


def add_noise(image):
    noisy_image = sk.util.random_noise(image, mode='gaussian', seed=None, clip=True)
    return noisy_image


def blur_image(image):
    kernel_size = KERNEL_SIZE_POSSIBILITIES[random.randint(0, len(KERNEL_SIZE_POSSIBILITIES) - 1)]
    sigma_x = random.randint(SIGMA_X_LOWER, SIGMA_X_UPPER)
    sigma_y = random.randint(SIGMA_Y_LOWER, SIGMA_Y_UPPER)
    return cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)


def apply_all_transforms(image):
    image = change_brightness(image)

    # 20% probability of blur
    if random.randint(1, 5) == 1:
        image = blur_image(image)

    # 20% probability of noise
    if random.randint(1, 5) == 1:
        image = add_noise(image)

    angle_change = np.random.uniform(ANGLE_CHANGE_MAX) - (ANGLE_CHANGE_MAX / 2)
    rows, columns, channels = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), angle_change, 1)

    translation_x = np.random.uniform(TRANSLATION_X_LOWER, TRANSLATION_X_UPPER)
    translation_y = np.random.uniform(TRANSLATION_Y_LOWER, TRANSLATION_Y_UPPER)
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + SHEAR_RANGE * np.random.uniform() - SHEAR_RANGE / 2
    pt2 = 20 + SHEAR_RANGE * np.random.uniform() - SHEAR_RANGE / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_matrix = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, rotation_matrix, (columns, rows))
    image = cv2.warpAffine(image, translation_matrix, (columns, rows))

    # 25% probability of shearing
    if random.randint(1, 4) == 1:
        image = cv2.warpAffine(image, shear_matrix, (columns, rows))

    return image


PROJECT_PATH = "/Users/mateuszdziubek/Desktop/BeerAI-Data"


def generate(amount_per_image, directory):
    images = []
    for image_name in os.listdir(directory):
        image_name = image_name.lower()
        if image_name.endswith(".jpg") or image_name.endswith(".jpeg") \
                or image_name.endswith(".png") or image_name.endswith(".bmp"):
            images.append(image_name)

    for image in images:
        for _ in range(amount_per_image):
            generated_image = apply_all_transforms(mpimg.imread(f"{directory}/{image}"))
            scipy.misc.toimage(generated_image).save(f"{directory}/generated{time.time()}.jpg")


def generate_for_all(amount_per_image, container_directory):
    for folder_name in os.listdir(container_directory):
        if not folder_name.startswith("."):
            generate(amount_per_image, f"{container_directory}/{folder_name}")


def save_labels(container_directory):
    labels = []
    for folder_name in os.listdir(container_directory):
        if not folder_name.startswith("."):
            labels.append(folder_name)
    file_path = f"{PROJECT_PATH}/data/labels/{str(datetime.datetime.now())}.txt"
    file = open(file_path, 'w')
    for label in labels:
        file.write("%s\n" % label)


save_labels(f"{PROJECT_PATH}/data/train")

# generate_for_all(150, f"{PROJECT_PATH}/data/train")

# generate(150, f"{PROJECT_PATH}/data/train/001harnas")
# generate(150, f"{PROJECT_PATH}/data/train/002kasztelan_niepaster")
# generate(150, f"{PROJECT_PATH}/data/train/003kasztelan_pszen")
# generate(150, f"{PROJECT_PATH}/data/train/004miloslaw_nieb")
# generate(150, f"{PROJECT_PATH}/data/train/005perla_chmiel")
# generate(150, f"{PROJECT_PATH}/data/train/006perla_export")
# generate(150, f"{PROJECT_PATH}/data/train/007somersby")
# generate(150, f"{PROJECT_PATH}/data/train/008warka")
# generate(150, f"{PROJECT_PATH}/data/train/009wojak")
# generate(150, f"{PROJECT_PATH}/data/train/010zywiec_bialy")
