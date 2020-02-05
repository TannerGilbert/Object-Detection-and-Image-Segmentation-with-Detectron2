# Object Detection and Instance Segmentation with Detecton2
![](https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png)

Detectron2 is Facebooks new library that implements state-of-the-art object detection algorithm. This repository shows you how to use Detectron2 for both inference as well as using transfer learning to train on your own data-set.

## Installation

See the official [installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Install using Docker

Another great way to install Detectron2 is by using Docker. Docker is great because you don't need to install anything locally, which allows you to keep your machine nice and clean.

If you want to run Detectron2 with Docker you can find a Dockerfile and docker-compose.yml file in the [docker directory of the repository]().

For those of you who also want to use Jupyter notebooks inside their container, I created a custom Docker configuration, which automatically starts Jupyter after running the container. If you're interested you can find the files in the [docker directory](https://github.com/facebookresearch/detectron2/tree/master/docker).

## Inference with pre-trained model

Using a pre-trained model for inference is as easy as loading in the cofiguration and weights and creating a predictor object. For a example check out [Detectron2_inference_with_pre_trained_model.ipynb](Detectron2_inference_with_pre_trained_model.ipynb).

## Training on a custom dataset

Training a model on a custom dataset is a bit more challenging because it requires that we [register our dataset](https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset). For a example showing you how to train a balloon detection model check out [Detectron2_train_on_a_custom_dataset.ipynb](Detectron2_train_on_a_custom_dataset.ipynb).

## Author
 **Gilbert Tanner**