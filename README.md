# phi3_vision_language
experiments with microsoft phi3 vision language model. Image captioning, OCR, data extraction

# Environment setting

The Phi-3-Vision-128K model uses flash attention, which requires certain types of GPU hardware to run. I test the model on a Nvidia A100 GPU


1) conda create -n llm_images python=3.10

2) conda activate llm_images

3) pip install torch==2.3.0 torchvision==0.18.0

4) pip install packaging

5) pip install pillow==10.3.0 chardet==5.2.0 flash_attn==2.5.8 accelerate==0.30.1 bitsandbytes==0.43.1 Requests==2.31.0 transformers==4.40.2 

6) pip uninstall jupyter

7) conda install -c anaconda jupyter

8) conda update jupyter

9) pip install --upgrade 'nbconvert>=7' 'mistune>=2'

10) pip install cchardet