import json
import wandb
from ultralytics import YOLO


def train_model(config=None):
    # Inicializa W&B con los parámetros actuales
    wandb.init(config=config)
    config = wandb.config

    # Acceso seguro a los parámetros necesarios
    project = config.get("project", "default_project")
    run_name = config.get("name", wandb.run.name)
    # devide 0 and 1
    devices_str  = "0,1"
    devices = [int(d) for d in devices_str.split(',')] if devices_str else None

    # Entrenar el modelo
    model = YOLO(config.model_weights)
    results = model.train(
        data=config.data,
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
        lr0=config.lr0,
        lrf=config.lrf,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        warmup_momentum=config.warmup_momentum,
        warmup_bias_lr=config.warmup_bias_lr,
        box=config.box,
        cls=config.cls,
        dfl=config.dfl,
        pose=config.pose,
        dropout=config.dropout,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        flipud=config.flipud,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        mixup=config.mixup,
        copy_paste=config.copy_paste,
        erasing=config.erasing,
        project=project,
        name=run_name,
        device=devices,
    )

    
    path_weights = f"{project}/{run_name}/weights/best.pt"
    print(f"Training completed. Best model weights saved at: {path_weights}")

    wandb.finish()


if __name__ == "__main__":
    import argparse

    # Argumento para el archivo sweep_config.json
    parser = argparse.ArgumentParser(description="Run a YOLO sweep using W&B.")
    parser.add_argument("--sweep_config", type=str, required=True, help="Path to the sweep_config.json file.")
    args = parser.parse_args()

    # Leer archivo sweep_config.json
    with open(args.sweep_config, "r") as f:
        sweep_config = json.load(f)

    # Crear y ejecutar el sweep
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["parameters"]["project"]["value"])
    wandb.agent(sweep_id, function=train_model)
