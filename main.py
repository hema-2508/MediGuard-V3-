"""
main.py
-------
Command-line entry point that wires together every module into the
full pipeline:

    python main.py train                 # train on DrugBankDDI, save best model
    python main.py evaluate              # evaluate best model on DrugBank test set
    python main.py external_validate     # evaluate best model on BioSNAPDDI
    python main.py predict --smiles1 ... --smiles2 ...   # score a single pair
    python main.py all                   # train -> evaluate -> external_validate

Data preparation (vocab, feature matrix, graph) is cached to disk on
first use so subsequent commands (evaluate / predict / external
validation) don't repeat expensive RDKit + graph-construction work.
Prediction uses a separate lightweight path that loads only cached
artifacts and the model checkpoint — no CSV reads.
"""

import argparse
import json
import os

import torch

from config import Config
from dataset import load_ddi_data, load_inference_artifacts
from graph_builder import build_graph
from model import DDIModel
from predict import DDIPredictor
from trainer import Trainer
from evaluate import evaluate_split
from utils import get_logger, set_seed, save_pickle, load_pickle, log_device_info

logger = get_logger("main")


def _resolve_device(args) -> torch.device:
    """CLI --device overrides the auto-detected GPU in config.py.
    Falls back to Config.DEVICE (cuda:0 if available, else cpu)."""
    if getattr(args, "device", None):
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"--device {args.device} requested but CUDA is not available on this "
                "machine; falling back to CPU."
            )
            device = torch.device("cpu")
    else:
        device = Config.DEVICE
    log_device_info(logger, device)
    return device


def _load_or_build_data():
    """Training/evaluation path: loads CSV splits, vocab, features, and graph.

    Reads DrugBank + BioSNAP CSVs and rebuilds the feature matrix. Use
    ``_load_inference_artifacts()`` for prediction instead.
    """
    data = load_ddi_data(
        train_csv=Config.DRUGBANK_TRAIN_CSV,
        test_csv=Config.DRUGBANK_TEST_CSV,
        biosnap_train_csv=Config.BIOSNAP_TRAIN_CSV,
        biosnap_test_csv=Config.BIOSNAP_TEST_CSV,
        val_split_ratio=Config.VAL_SPLIT_RATIO,
        use_cache=True,
    )

    if os.path.exists(Config.GRAPH_CACHE_PATH):
        logger.info(f"Loading cached graph from {Config.GRAPH_CACHE_PATH}")
        graph = torch.load(Config.GRAPH_CACHE_PATH, weights_only=False)
    else:
        graph = build_graph(data.feature_matrix, data.train)
        torch.save(graph, Config.GRAPH_CACHE_PATH)
        logger.info(f"Saved graph to {Config.GRAPH_CACHE_PATH}")

    return data, graph


def _load_inference_artifacts():
    """Inference-only path: cached vocab + graph, no CSV I/O."""
    return load_inference_artifacts()


def cmd_train(args):
    device = _resolve_device(args)
    set_seed(Config.SEED)
    data, graph = _load_or_build_data()

    model = DDIModel(
        in_channels=Config.NODE_FEATURE_DIM,
        hidden_channels=Config.SAGE_HIDDEN_DIM,
        num_layers=Config.SAGE_NUM_LAYERS,
        sage_dropout=Config.SAGE_DROPOUT,
        pair_hidden_dim=Config.PAIR_HIDDEN_DIM,
        pair_dropout=Config.PAIR_DROPOUT,
    )

    trainer = Trainer(model=model, graph=graph, train_pairs=data.train, val_pairs=data.val, device=device)
    result = trainer.fit(epochs=args.epochs)

    history_path = os.path.join(Config.RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    logger.info(f"Saved training history to {history_path}")


def _load_best_model(graph, device: torch.device = Config.DEVICE) -> DDIModel:
    ckpt = torch.load(Config.BEST_MODEL_PATH, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {})
    model = DDIModel(
        in_channels=model_cfg.get("in_channels", Config.NODE_FEATURE_DIM),
        hidden_channels=model_cfg.get("hidden_channels", Config.SAGE_HIDDEN_DIM),
        num_layers=model_cfg.get("num_layers", Config.SAGE_NUM_LAYERS),
        sage_dropout=model_cfg.get("sage_dropout", Config.SAGE_DROPOUT),
        pair_hidden_dim=model_cfg.get("pair_hidden_dim", Config.PAIR_HIDDEN_DIM),
        pair_dropout=model_cfg.get("pair_dropout", Config.PAIR_DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def cmd_evaluate(args):
    device = _resolve_device(args)
    data, graph = _load_or_build_data()
    model = _load_best_model(graph, device=device)
    save_path = os.path.join(Config.RESULTS_DIR, "drugbank_test_report.json")
    evaluate_split(model, graph, data.test, split_name="DrugBank Test Set", device=device, save_path=save_path)


def cmd_external_validate(args):
    device = _resolve_device(args)
    data, graph = _load_or_build_data()
    if data.external is None or len(data.external) == 0:
        logger.error(
            f"Expected:\n"
            f"{Config.BIOSNAP_TRAIN_CSV}\n"
            f"{Config.BIOSNAP_TEST_CSV}"
        )
        return
    model = _load_best_model(graph, device=device)
    save_path = os.path.join(Config.RESULTS_DIR, "biosnap_external_report.json")
    evaluate_split(
        model, graph, data.external, split_name="BioSNAP External Validation",
        device=device, save_path=save_path,
    )


def cmd_predict(args):
    device = _resolve_device(args)
    vocab, graph = _load_inference_artifacts()
    predictor = DDIPredictor(
        checkpoint_path=Config.BEST_MODEL_PATH,
        graph=graph,
        vocab=vocab,
        device=device,
    )
    result = predictor.predict_pair(args.smiles1, args.smiles2)
    print(json.dumps(result, indent=2))


def cmd_all(args):
    cmd_train(args)
    cmd_evaluate(args)
    cmd_external_validate(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphSAGE Drug-Drug Interaction (DDI) pipeline")

    # Shared --device flag available on every subcommand, e.g.:
    #   python main.py train --device cuda:0
    #   python main.py evaluate --device cpu
    device_parent = argparse.ArgumentParser(add_help=False)
    device_parent.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda:0', 'cuda', or 'cpu'. "
        "Defaults to the auto-detected GPU (Config.DEVICE) if available.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", parents=[device_parent], help="Train the model on DrugBankDDI")
    p_train.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", parents=[device_parent], help="Evaluate best model on DrugBank test set")
    p_eval.set_defaults(func=cmd_evaluate)

    p_ext = sub.add_parser("external_validate", parents=[device_parent], help="Evaluate best model on BioSNAPDDI")
    p_ext.set_defaults(func=cmd_external_validate)

    p_pred = sub.add_parser(
        "predict", parents=[device_parent], help="Predict interaction probability for a single drug pair"
    )
    p_pred.add_argument("--smiles1", type=str, required=True)
    p_pred.add_argument("--smiles2", type=str, required=True)
    p_pred.set_defaults(func=cmd_predict)

    p_all = sub.add_parser(
        "all", parents=[device_parent], help="Run train -> evaluate -> external_validate in sequence"
    )
    p_all.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()