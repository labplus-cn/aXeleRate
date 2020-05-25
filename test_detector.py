from axelerate import setup_training, setup_inference

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()


detector_config = {
    "model": {
        "type":                 "Detector",
        "architecture":         "MobileNet2_5",
        "input_size":           224,
        "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "labels":               ["raccoon"],
        "coord_scale": 		1.0,
        "class_scale": 		1.0,
        "object_scale": 		5.0,
        "no_object_scale": 	1.0
    },
    "pretrained": {
        "full":   				""
    },
    "weights": {
        "full":   				"",
        "backend":   		    "imagenet",
        "save_bottleneck":      False
    },

    "train": {
        "actual_epoch":         50,
        "train_image_folder":   "sample_datasets/detector_raccoon/imgs",
        "train_annot_folder":   "sample_datasets/detector_raccoon/anns",
        "train_times":          1,
        "valid_image_folder":   "sample_datasets/detector_raccoon/imgs_validation",
        "valid_annot_folder":   "sample_datasets/detector_raccoon/anns_validation",
        "valid_times":          1,
        "valid_metric":         "mAP",
        "batch_size":           32,
        "learning_rate":        1e-3,
        "saved_folder":   		"results/detector_raccoon",
        "first_trainable_layer": "",
        "augumentation":				True,
        "is_only_detect": 		False
    },
    "converter": {
        "type":   				["k210", "tflite"]
    }
}


# 使用字典传参
model_path = setup_training(config_dict=detector_config)
# 使用json传参
# model_path = setup_training(config_file='configs/detector_raccoon.json')

setup_inference(detector_config, model_path)
