from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse
import os, glob


def get_model(model_path, config_path, threshold):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path

    return DefaultPredictor(cfg), cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects from images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    parser.add_argument('-i', '--image_path', type=str, default='', help='Path to image (or folder of images)')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='', help= 'Path to save the output. If None output won\'t be saved')
    args = parser.parse_args()

    # Create output folder if save_path variable is not none
    if args.save_path:
        os.makedirs(args.ave_path, exist_ok=True)

    predictor, cfg = get_model(args.model, args.config, args.threshold)

    if os.path.isdir(args.image_path):
        image_paths = []
        for file_extension in ('*.png', '*jpg'):
            image_paths.extend(glob.glob(os.path.join(args.image_path, file_extension)))
    else:
        image_paths = [args.image_path]
    
    for i_path in image_paths:
        image = cv2.imread(i_path)
        
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        if args.show:
            cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        if args.save_path:
            cv2.imwrite(os.path.join(args.save_path, os.path.basename(i_path)), v.get_image()[:, :, ::-1])
    cv2.destroyAllWindows()