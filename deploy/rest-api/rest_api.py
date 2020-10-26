from flask import Flask, request, jsonify 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np
import argparse
import requests


app = Flask(__name__)


def get_model(model_path, config_path, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = 'cpu'

    return DefaultPredictor(cfg), cfg


@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.json["imageUrl"]
    image = cv2.imdecode(np.frombuffer(requests.get(image_url).content, np.uint8), cv2.IMREAD_COLOR)

    pred = predictor(image)

    instances = pred["instances"]
    scores = instances.get_fields()["scores"].tolist()
    pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
    }

    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects from webcam images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    args = parser.parse_args()

    predictor, cfg = get_model(args.model, args.config, args.threshold)

    app.run(port=3000, debug=True)