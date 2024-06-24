import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def detect_and_compute_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


# Metrics and plots
def plot_matches(img1, img2, kp1, kp2, matches, path_save_plot):
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_image)
    plt.title("SIFT Feature Matching - number matches " + str(len(matches)))
    plt.savefig(path_save_plot)


def get_matched_points(keypoints1, keypoints2, matches):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2

def compute_homography(points1, points2):
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    return h

def warp_image(image, h, shape):
    height, width = shape[:2]
    warped_image = cv2.warpPerspective(image, h, (width, height))
    return warped_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_standard_image', default="resources/cie_fronte_standard.jpg", type=str)
    parser.add_argument('--dir_aligned_images', default="resources/aligned_sift", type=str)
    parser.add_argument('--path_modified_image', type=str)

    opt = parser.parse_args()
    path_standard_image = opt.path_standard_image
    dir_aligned_images = opt.dir_aligned_images
    path_modified_image = opt.path_modified_image

    file_mod_image = path_modified_image.split("/")[-1]
    name_mod_image = file_mod_image.split(".")[0]
    path_save_plot = os.path.join(dir_aligned_images, name_mod_image + "_matches.png")
    path_save_aligned_image = os.path.join(dir_aligned_images, file_mod_image)

    # Caricamento delle immagini
    image1 = load_image(path_standard_image)
    image2 = load_image(path_modified_image)

    # Rilevamento dei keypoints e descrittori
    keypoints1, descriptors1 = detect_and_compute_keypoints(image1)
    keypoints2, descriptors2 = detect_and_compute_keypoints(image2)

    # Matching dei keypoints
    matches = match_keypoints(descriptors1, descriptors2)

    print(f"Number of keypoints in the original image: {len(keypoints1)}")
    print(f"Number of keypoints in the rotated image: {len(keypoints2)}")
    print(f"Number of good matches: {len(matches)}")

    # Plot matching
    plot_matches(image1, image2, keypoints1, keypoints2, matches, path_save_plot)

    # Estrazione dei punti corrispondenti
    points1, points2 = get_matched_points(keypoints1, keypoints2, matches)

    # Calcolo dell'omografia
    h = compute_homography(points1, points2)

    # Rotazione e trasformazione della seconda immagine
    aligned_image2 = warp_image(image2, h, image1.shape)

    # Salvataggio dell'immagine allineata
    cv2.imwrite(path_save_aligned_image, aligned_image2)