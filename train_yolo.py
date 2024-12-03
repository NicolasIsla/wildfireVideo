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

    # Entrenar el modelo
    model = YOLO(config.model_weights)
    results = model.train(
        data=config.data,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        lr0=config.lr0,
        momentum=config.momentum,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        shear=config.shear,
        perspective=config.perspective,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        mixup=config.mixup,
        autoaugment=config.auto_augment,  # Sólo si 'autoaugment' está soportado por Ultralytics
        project=project,
        name=run_name,
    )

    # Log de resultados
    wandb.log({"val_loss": results["metrics/val_loss"]})
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
