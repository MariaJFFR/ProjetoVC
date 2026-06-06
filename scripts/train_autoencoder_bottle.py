import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from autoencoder import Autoencoder
from bottle_protocol import bottle_train_good_paths, split_train_validation_good
from dataset import criar_loader_de_paths
from treino import treinar_autoencoder_com_validacao
from utils import get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bottle autoencoder on normal images only.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    all_good = bottle_train_good_paths()
    train_paths, val_paths = split_train_validation_good(
        all_good,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_loader = criar_loader_de_paths(
        train_paths,
        batch_size=args.batch_size,
        tamanho_imagem=args.image_size,
        shuffle=True,
    )
    val_loader = criar_loader_de_paths(
        val_paths,
        batch_size=args.batch_size,
        tamanho_imagem=args.image_size,
        shuffle=False,
    )

    model_path = ROOT / "models" / "autoencoder_bottle_best.pth"
    modelo = Autoencoder()
    history = treinar_autoencoder_com_validacao(
        modelo=modelo,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        caminho_modelo=str(model_path),
    )

    save_json(
        {
            "category": "bottle",
            "seed": args.seed,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "train_normal_count": len(train_paths),
            "validation_normal_count": len(val_paths),
            "model_path": str(model_path),
            "history": history,
        },
        ROOT / "results" / "autoencoder_bottle_training.json",
    )

    print(f"Saved model: {model_path}")
    print(f"Saved training history: {ROOT / 'results' / 'autoencoder_bottle_training.json'}")


if __name__ == "__main__":
    main()
