class Config:
    # Image dimensions
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3

    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4  
    T_MAX = 50  
    ETA_MIN = 1e-6  

    # Dataset paths
    # LABELED_DATA_PATH = ""
    LABELED_DATA_PATH = r""
    # UNLABELED_DATA_PATH = ""

    # Model save path
    MODEL_SAVE_PATH = ""

    # # ViT parameters
    # PRETRAINED_VIT = ""
    # Hyperparameter combinations
    # HYPERPARAMETER_COMBINATIONS = [
    #     {
    #         "LEARNING_RATE": [1e-3, 5e-4, 1e-4, 1e-5],
    #         "BATCH_SIZE": [16, 32],
    #         "WEIGHT_DECAY": [5e-4, 1e-4, 1e-5],
    #         "T_MAX": [25, 50, 100],
    #     }
    # ]

config = Config()
