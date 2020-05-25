from axelerate import setup_training, setup_inference

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()


classifler_config = {
    "model": {
        "type":                 "Classifier",
        "architecture":         "MobileNet2_5",
        "input_size":           224,
        "fully-connected":      [100, 50],
        "labels":               [],
        "dropout": 		0.5
    },

    "weights": {
        "full":   				"",
        "backend":   		    "imagenet",
        "save_bottleneck":      False
    },
    "pretrained": {
        "full":   				""
    },
    "train": {
        "actual_epoch":         2,
        "train_image_folder":   "sample_datasets/classifier_flower/imgs",
        "train_times":          4,
        "valid_image_folder":   "sample_datasets/classifier_flower/imgs_validation",
        "valid_times":          4,
        "valid_metric":         "val_accuracy",
        "batch_size":           10,
        "learning_rate":        1e-4,
        "saved_folder":   		"results/classifier_flower",
        "first_trainable_layer": "",
        "augumentation":				True
    },
    "converter": {
        "type":   				["k210", "tflite"]
    }

}

# 使用字典传参
model_path = setup_training(config_dict=classifler_config)
# 使用json传参
# model_path = setup_training(config_file='configs/classifier_flower.json')

setup_inference(classifler_config, model_path)
