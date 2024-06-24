import numpy as np
import cv2
import argparse
import os

threshold = 100

'https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/'


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.Laplacian(gray, cv2.CV_64F).var()


def unblur(image):

	# Create the sharpening kernel
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	# Sharpen the image
	unblur_image = cv2.filter2D(image, -1, kernel)

	return unblur_image


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--path_blur_image', type=str)
	parser.add_argument('--dir_unblurred_image', default="resources/aligned_sift_unblurred", type=str)

	opt = parser.parse_args()
	path_blur_image = opt.path_blur_image
	dir_unblurred_image = opt.dir_unblurred_image
	file_image = path_blur_image.split("/")[-1]

	# load the image
	image = cv2.imread(path_blur_image)

	#calculate the fm for the first time
	fm = variance_of_laplacian(image)

	# loop until fm is bigger then the threshold
	iter = 0
	while fm < threshold:

		# unblur the image
		image = unblur(image)
		iter += 1

		# calculate the focus measure of the new image
		fm = variance_of_laplacian(image)

	# Save the image
	print("iter: {} - fm: {}".format(iter, fm))
	path_unblurred_image = os.path.join(dir_unblurred_image,file_image)
	cv2.imwrite(path_unblurred_image, image)





