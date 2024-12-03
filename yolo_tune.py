import json
import wandb
# from wandb.integration.ultralytics import add_wandb_callback
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
    # add_wandb_callback(model)
    model.tune(
        data=config.data,
        epochs=config.epochs,
        iterations=500,
        
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

    
    train_model(sweep_config)