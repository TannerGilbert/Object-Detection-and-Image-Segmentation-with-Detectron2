from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse


def get_model(model_path, config_path, threshold):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path

    return DefaultPredictor(cfg), cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects from webcam images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='', help= 'Path to save the output. If None output won\'t be saved')
    args = parser.parse_args()

    predictor, cfg = get_model(args.model, args.config, args.threshold)

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    if args.save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (width, height))
        
    while cap.isOpened():
        ret, image = cap.read()
        
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        if args.show:
            cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        if args.save_path:
            out.write(v.get_image()[:, :, ::-1])
    cap.release()
    if args.save_path:
        out.release()
    cv2.destroyAllWindows()