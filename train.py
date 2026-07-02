import argparse
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train YOLO-DFA with the paper training protocol.")

    parser.add_argument("--model", type=str, default=str(root / "Model" / "YOLO-DFA.yaml"))
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0002)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--cos-lr", type=str2bool, default=True)

    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--project", type=str, default=str(root / "runs" / "train"))
    parser.add_argument("--name", type=str, default="YOLO-DFA")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cache", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    model_source = args.resume if args.resume else args.model
    model_path = Path(model_source)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file or checkpoint not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed or not available in the current environment. "
            "Install the required environment first, then rerun this script."
        ) from exc

    train_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "amp": args.amp,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }

    if args.cache is not None:
        train_kwargs["cache"] = args.cache

    if args.resume:
        train_kwargs["resume"] = True

    try:
        model = YOLO(str(model_path))
        model.train(**train_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Training failed. If the error says that DynamicConv, C2f_DynamicConv, "
            "C2f_Bifocal, DK_FMM, iRMB_Zoom, or SSEM is not defined, register the "
            "custom modules in the active Ultralytics model parser before training."
        ) from exc


if __name__ == "__main__":
    main()
