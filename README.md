# Object Detection and Instance Segmentation with Detecton2
![](https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png)

Detectron2 is Facebooks new library that implements state-of-the-art object detection algorithm. This repository shows you how to use Detectron2 for both inference as well as using transfer learning to train on your own data-set.

## Installation

See the official [installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Install using Docker

Another great way to install Detectron2 is by using Docker. Docker is great because you don't need to install anything locally, which allows you to keep your machine nice and clean.

If you want to run Detectron2 with Docker you can find a Dockerfile and docker-compose.yml file in the [docker directory of the repository](https://github.com/facebookresearch/detectron2/tree/master/docker).

For those of you who also want to use Jupyter notebooks inside their container, I created a custom Docker configuration, which automatically starts Jupyter after running the container. If you're interested you can find the files in the [docker directory](https://github.com/TannerGilbert/Object-Detection-and-Image-Segmentation-with-Detectron2/tree/master/docker).

## Inference with pre-trained model

* [Detectron2 Inference with pretrained model](Detectron2_inference_with_pre_trained_model.ipynb).
* [Detect from Webcam or Video](detect_from_webcam_or_video.py)
* [Detectron2 RestAPI](deploy/rest-api)

## Training on a custom dataset

* [Detectron2 train on a custom dataset](Detectron2_train_on_a_custom_dataset.ipynb).
* [Detectron2 train with data augmentation](Detectron2_Train_on_a_custom_dataset_with_data_augmentation.ipynb)
* [Detectron2 Chess Detection](Detectron2_Detect_Chess_Detection.ipynb)
* [Detectron2 Vehicle Detection](Detectron2_Vehicle_Detection.ipynb)

## Export model

* [Detectron2 Export model to Caffe2, Onnx and Torchvision](Detectron2_train_and_export_model.ipynb)

## Author
 **Gilbert Tanner**
