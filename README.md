# ML-Face-Recognition

## Overview
Computer Vision coursework submission for my MSc Data Science @ City University, London (2020).

**Aim:** Create end-to-end ML-based face recognition system, capable of identifying students from my class in any individual or group image.

## Repo contents
* code - implementation of face recognition system in python
* font - font type file used to print student's unique ID's in images when they are identified by the system
* report - report outlining theory and implementation approach in more detail, and analysing key findings
* video - video showing face recognition system in action.

## Code contents

The python code consists of the below files, to perform each part of the analysis pipeline (located in the code subfolder).

Python files:
* extract_faces.ipynb - extract faces from the training images
* train_models.ipynb - train the models using the extracted faces
* RecogniseFace.ipynb - use the RecogniseFace function to run the different models on the test images
* utils.py - contains the code for the RecogniseFace function and other key tasks (e.g. extracting features from images, visualising model predictions, etc)

## Setting up environment to run the code

1. Unzip my coursework submission folder (containing my code) on your laptop; somewhere easily accessible from your command line (e.g. your desktop)
2. Download and install the anaconda python distribution (if you haven't done so already) using the following link: https://www.anaconda.com/distribution/
3. Make sure to download the Python 3.7 version, and select the right option for your operating system (Windows, macOS, Linux)
4. Open your command line (Terminal for Mac, Anaconda Command Prompt for Windows), and navigate to the code subfolder in the coursework submission folder, using this command:

```
cd <coursework submission folder address>/code 
```
5. In the command line, run the following commands to create a virtual environment with all the necessary packages to run the python code: 

```
# create new environment
conda env create -f environment.yml;
# activate the environment
conda activate Cvcoursework2020;
# delete opencv-python and install opencv-contrib-python
pip uninstall opencv-python -y;
pip install opencv-contrib-python==3.4.2.17 --force-reinstall;
```
6. In the command line, run the following command to open jupyter notebook:

```
jupyter notebook
```
7. You can now open and execute the .ipynb files (containing the python code) to perform each part of the pipeline as required.

## Running the code

To replicate the full pipeline end-to-end, run the .ipynb files in this order:
1. extract_faces.ipynb 
2. train_models.ipynb
3. RecogniseFace.ipynb

Make sure to apply the below configuration changes to the files before running them.
 
extract_faces.ipynb: 
* In the first cell, set images_i_path to the folder path containing your individual training images (i.e. training images consisting of a single person)
* In the first cell, set images_g_path to the folder path containing your group training images (i.e. training images consisting of multiple people)
* In the first cell, set faces_path to the folder path where you'd like to save the extracted faces.

train_models.ipynb:
* In the first cell, set faces_path to the folder path containing the extracted faces for training.
* In the first cell, set models_path to the folder path where you'd like to save the trained models.
