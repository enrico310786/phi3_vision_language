# phi3_vision_language
Experiments with microsoft phi3 vision language model. Image captioning, OCR, data extraction

# Environment setting

The Phi-3-Vision-128K model uses flash attention, which requires certain types of GPU hardware to run. I test the model on a Nvidia A100 GPU


1) conda create -n llm_images python=3.10

2) conda activate llm_images

3) pip install torch==2.3.0 torchvision==0.18.0

4) pip install packaging

5) pip install pillow==10.3.0 chardet==5.2.0 flash_attn==2.5.8 accelerate==0.30.1 bitsandbytes==0.43.1 Requests==2.31.0 transformers==4.40.2 albumentations==1.3.1 opencv-contrib-python==4.10.0.84 matplotlib==3.9.0

6) pip uninstall jupyter

7) conda install -c anaconda jupyter

8) conda update jupyter

9) pip install --upgrade 'nbconvert>=7' 'mistune>=2'

10) pip install cchardet

# Notebooks

The notebook Phi-3 Vision-Language Model_v1.ipynb show the application of the Phi3 model on documents 
as the Italian identity card, the italian driving licence and the italian health insurance card.
The application of this notebook is told on the Medium story
https://medium.com/@enrico.randellini/exploring-microsoft-phi3-vision-language-model-as-ocr-for-document-data-extraction-c269f7694d62

The notebook Phi-3 Vision-Language Model_v2.ipynb show the application of the Phi3 model on
modified images of the italian identity card. It shows that, in order to extract the correct data, 
the images have to be aligned and readable. Thus, I apply some non deep learning computer vision techniques
to restore the quality of the images as SIFT, blur, contrast and brightness corrections.

The application of this notebook is told on the Medium story


# Image transformations

The script image_modifier.py allows you to take an image of a document and rotate, scale, blur it as well as change 
its contrast and brightness. Thi is useful to simulate scanned and photocopied document

```bash
python image_modifier.py --path_original_image <path to the original image to modify> --path_modified_image <path where to save the transformed image> --to_gray <true or false>
```

The scrip apply_sift.py, apply the SIFT algorithm to the reference and test image. Thus, once determined the common keypoints,
it rotates and scales the test images to align it with the reference one

```bash
python apply_sift.py --path_standard_image <path to the reference image> --dir_aligned_images <directory where to save the aligned images> --path_modified_image <path of the modified image to align>
```

The script apply_blur_detection_remotion.py, detect if the image is blurred and remove it

```bash
python apply_blur_detection_remotion.py --path_blur_image <path of the blurred image> --dir_unblurred_image <directory where to save the unblurred images>
```

The script apply_contrast_brightness_shift.py, changes the contrast and brightness of the modified image and aligns them to those of the reference image

```bash
python apply_contrast_brightness_shift.py --path_test_image <path of the modified image to correct> --path_standard_image <path of the reference image> --dir_clean_image <directory where to store the corrected images>
```

On the directory "resources/transformed" are stored the transformed images obtained with the script image_modifier.py.

On the directory "resources/aligned_sift" are stored the images obtained from those transformed and aligned with the SIFT algorithm using the script apply_sift.py

On the directory "resources/aligned_sift_unblurred" are stored the images that, after aligned with SIFT are unblurred with the script apply_blur_detection_remotion.py

On the directory "resources/cleaned_images" are stored the final cleaned images where, after the blur removal, are also aligned with the reference image in terms of contrast and brightness