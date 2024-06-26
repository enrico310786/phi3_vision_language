import cv2
import albumentations as A
import argparse


transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
    A.RandomGamma(p=1),
    A.ShiftScaleRotate(shift_limit=0.065, scale_limit=0.7, rotate_limit=90, p=1),
    A.Blur(blur_limit=7, p=1),
])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_original_image', type=str)
    parser.add_argument('--path_modified_image', type=str)
    parser.add_argument('--to_gray', type=str2bool, default='false')

    opt = parser.parse_args()
    path_original_image = opt.path_original_image
    path_modified_image = opt.path_modified_image
    to_gray = opt.to_gray

    image = cv2.imread(path_original_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # add padding
    image = cv2.copyMakeBorder(image, top=350, bottom=350, left=200, right=200, borderType=0)

    # add transformations
    modified_image = transform(image=image)['image']
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)

    if to_gray:
        modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

    # save the image
    cv2.imwrite(path_modified_image, modified_image)
