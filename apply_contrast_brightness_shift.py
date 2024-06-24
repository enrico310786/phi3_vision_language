import cv2
import numpy as np
import argparse
import os


def calculate_brightness_and_contrast(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate brightness (mean)
    brightness = np.mean(gray)
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    return brightness, contrast


def adjust_brightness_and_contrast(image, target_brightness, target_contrast):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate current brightness and contrast
    current_brightness, current_contrast = calculate_brightness_and_contrast(image)

    # Calculate scaling factor and offset
    if current_contrast != 0:
        scaling_factor = target_contrast / current_contrast
    else:
        scaling_factor = 1

    offset = target_brightness - scaling_factor * current_brightness

    # Adjust brightness and contrast
    adjusted_image = cv2.convertScaleAbs(image, alpha=scaling_factor, beta=offset)

    return adjusted_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_test_image', type=str)
    parser.add_argument('--path_standard_image', default="resources/cie_fronte_standard.jpg", type=str)
    parser.add_argument('--dir_clean_image', default="resources/cleaned_images", type=str)

    opt = parser.parse_args()
    path_test_image = opt.path_test_image
    path_standard_image = opt.path_standard_image
    dir_clean_image = opt.dir_clean_image
    file_image = path_test_image.split("/")[-1]

    # Load the images
    standard_image = cv2.imread(path_standard_image)
    image_to_adjust = cv2.imread(path_test_image)

    # Calculate brightness and contrast of the standard image
    standard_brightness, standard_contrast = calculate_brightness_and_contrast(standard_image)

    # Adjust the second image
    adjusted_image = adjust_brightness_and_contrast(image_to_adjust, standard_brightness, standard_contrast)

    # Save the adjusted image
    cv2.imwrite(os.path.join(dir_clean_image, file_image), adjusted_image)