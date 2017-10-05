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

    image = cv2.warpAffine(image, rotation_matrix, (columns, rows))
    image = cv2.warpAffine(image, translation_matrix, (columns, rows))

    return image


def generate(amount_per_image, directory=None):
    images = []
    for image_name in os.listdir(directory):
        image_name = image_name.lower()
        if image_name.endswith(".jpg") or image_name.endswith(".jpeg") \
                or image_name.endswith(".png") or image_name.endswith(".bmp"):
            images.append(image_name)

    for image in images:
        for _ in range(amount_per_image):
            if directory is not None:
                generated_image = apply_all_transforms(mpimg.imread(f"{directory}/{image}"))
                scipy.misc.toimage(generated_image).save(f"{directory}/generated{time.time()}.jpg")
            else:
                generated_image = apply_all_transforms(mpimg.imread(image))
                scipy.misc.toimage(generated_image).save(f"generated{time.time()}.jpg")
