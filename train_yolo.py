import wandb
from ultralytics import YOLO

def train_model(config=None):
    # Inicializa W&B con los parámetros actuales
    wandb.init(config=config)
    config = wandb.config

    # Cargar el modelo YOLO
    model = YOLO(config.model_weights)

    # Configurar augmentaciones
    augmentations = {
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "shear": config.shear,
        "perspective": config.perspective,
        "fliplr": config.fliplr,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "auto_augment": config.auto_augment,
    }

    # Entrenar el modelo
    results = model.train(
        data=config.data,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        lr0=config.lr0,
        momentum=config.momentum,
        augmentations=augmentations,  # Pasar augmentaciones
        project=config.project,
        name=config.name,
    )

    # Log de resultados
    wandb.log({"val_loss": results["metrics/val_loss"]})
    wandb.finish()

if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "model_weights": {"value": "yolov5s.pt"},
            "data": {"value": "/data/nisla/DS_08_V1/DS/data.yaml"},
            "epochs": {"values": [1, 2, 3]},
            "batch": {"values": [8, 16, 32, 64]},
            "lr0": {"min": 0.0001, "max": 0.01},
            "momentum": {"min": 0.8, "max": 0.99},
            "imgsz": {"values": [416, 512, 640]},
            "hsv_h": {"min": 0.0, "max": 0.1},
            "hsv_s": {"min": 0.0, "max": 1.0},
            "hsv_v": {"min": 0.0, "max": 1.0},
            "degrees": {"min": -15, "max": 15},
            "translate": {"min": 0.0, "max": 0.2},
            "scale": {"min": 0.5, "max": 1.5},
            "shear": {"min": 0.0, "max": 10.0},
            "perspective": {"min": 0.0, "max": 0.001},
            "fliplr": {"min": 0.0, "max": 1.0},
            "mosaic": {"min": 0.0, "max": 1.0},
            "mixup": {"min": 0.0, "max": 1.0},
            "auto_augment": {"values": ["randaugment", "autoaugment", "augmix"]},
        },
    }

    # Inicia el sweep
    sweep_id = wandb.sweep(sweep_config, project="yolo_augmentation_sweep")
    wandb.agent(sweep_id, function=train_model)
