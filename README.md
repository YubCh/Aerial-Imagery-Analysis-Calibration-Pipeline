# Aerial-Imagery-Analysis-Calibration-Pipeline

This Project is a simple data-analysis pipeline for aerial drone imagery. It ingests a dataset of aerial images, computes statistical metrics, detects anomalies, goes through deep learning feature, groups and searches images by visual similarities, normalizes exposures and finally presents the output in a dashboard.

Built on the VisDrone aerial imagery dataset(around 6500 images)


# What it does
*Statistical analysis & data mining: computes brightness, contrast, and blur metrics per image. Detects anomalies in images using z-score with Isolation Forest.

*Deep-learning features: we attract 512-dimensional embeddings from every image using the pretrained model: ResNet18 with IMAGENET1K_V1 as weight. While ignorning the final layer of resnet18

*Clustering & similarity search: groups embedded images into n clusters using the kmeans algorithm to find visually similar images

*Camera & image calibration: corrects lens distortion (checkboard calibration - still working on it) and normalizes uneven exposures with the CLAHE image processing aglorithm

*Database & Dashboard: stores all results in SQLite database and presents them through a Streamlit dashboard

# Architecture
The pipeline is organized into 6stages, with data flowing from [S1] ingestion through [S2 - S4] analysis into [S5] storage and [S6] presentation

                                [S2] Statistic
Raw Images -> [S1] Ingestion -> [S3] Deep learning -> [S5] Database -> [S6] Dashboard
                                [S4] Calibration
                                

#Key Results
                        

# Tech Stack
Python, OpenCV, Pytorch(ResNet18), scikit-learn, pandas, NumPy, SQLAlchemy + SQLite, Streamlit


# Project Structure








# How to Run

## Installation

```bash
``` 



# Limitations & future work




## Data sources & acknowledgements

- **VisDrone dataset** — aerial imagery used throughout this project. Provided by the
  AISKYEYE team at the Lab of Machine Learning and Data Mining, Tianjin University.
  Project: https://github.com/VisDrone/VisDrone-Dataset

- **Camera calibration reference images** — checkerboard images from
  paulmelis/opencv-camera-calibration (https://github.com/paulmelis/opencv-camera-calibration),
  used to demonstrate lens distortion correction.