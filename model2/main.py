"""
main.py
-------
Command-line entry point for Model-2 medicine extraction:

    python main.py train
    python main.py evaluate
    python main.py predict --text "Tab. Metformin 500 mg 1 tab BD x 30 days PO"
    python main.py serve
    python main.py all
"""

import argparse
import json
import os

import torch
import uvicorn
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import Config
from dataset import load_ner_data
from evaluate import evaluate_model, save_evaluation_report
from inference import MedicineExtractor
from trainer import build_trainer
from utils import get_logger, log_device_info, save_json, set_seed

logger = get_logger("main")


def _resolve_device(args) -> torch.device:
    if getattr(args, "device", None):
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"--device {args.device} requested but CUDA is not available; falling back to CPU."
            )
            device = torch.device("cpu")
    else:
        device = Config.DEVICE
    log_device_info(logger, device)
    return device


def cmd_train(args) -> None:
    device = _resolve_device(args)
    set_seed(Config.SEED)
    train_split, val_split, _ = load_ner_data()
    trainer = build_trainer(train_split, val_split, device=device)
    result = trainer.fit(epochs=args.epochs)

    history_path = os.path.join(Config.RESULTS_DIR, "training_history.json")
    save_json(result["history"], history_path)
    logger.info(f"Saved training history to {history_path}")


def _load_eval_model(checkpoint_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()
    return model, tokenizer


def cmd_evaluate(args) -> None:
    device = _resolve_device(args)
    _, _, test_split = load_ner_data()
    model, tokenizer = _load_eval_model(args.checkpoint, device)
    metrics = evaluate_model(model, test_split, tokenizer, device=device)
    save_evaluation_report(metrics, split_name="Test Set")


def cmd_predict(args) -> None:
    _resolve_device(args)
    extractor = MedicineExtractor(checkpoint_path=args.checkpoint, device=Config.DEVICE)

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            if args.input_file.endswith(".json"):
                payload = json.load(handle)
                if isinstance(payload, list):
                    texts = payload
                elif isinstance(payload, dict) and "texts" in payload:
                    texts = payload["texts"]
                else:
                    raise ValueError("JSON input file must be a list of strings or {'texts': [...]}")
                results = extractor.extract_batch(texts)
            else:
                results = [extractor.extract(handle.read())]
    else:
        results = [extractor.extract(args.text)]

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


def cmd_serve(args) -> None:
    _resolve_device(args)
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


def cmd_all(args) -> None:
    cmd_train(args)
    cmd_evaluate(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MediGuard Model-2: Medicine Extraction")

    device_parent = argparse.ArgumentParser(add_help=False)
    device_parent.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda:0' or 'cpu'. Defaults to Config.DEVICE.",
    )

    checkpoint_parent = argparse.ArgumentParser(add_help=False)
    checkpoint_parent.add_argument(
        "--checkpoint",
        type=str,
        default=Config.BEST_MODEL_PATH,
        help="Path to a saved model checkpoint directory.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", parents=[device_parent], help="Fine-tune BioBERT on prescription NER")
    p_train.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser(
        "evaluate",
        parents=[device_parent, checkpoint_parent],
        help="Evaluate a checkpoint on the test set",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser(
        "predict",
        parents=[device_parent, checkpoint_parent],
        help="Extract medicines from prescription text",
    )
    p_pred.add_argument("--text", type=str, default=None, help="Single prescription / OCR text")
    p_pred.add_argument("--input-file", type=str, default=None, help="Text or JSON file with batch input")
    p_pred.set_defaults(func=cmd_predict)

    p_serve = sub.add_parser("serve", parents=[device_parent], help="Start the FastAPI inference server")
    p_serve.add_argument("--host", type=str, default=Config.API_HOST)
    p_serve.add_argument("--port", type=int, default=Config.API_PORT)
    p_serve.set_defaults(func=cmd_serve)

    p_all = sub.add_parser("all", parents=[device_parent], help="Train then evaluate")
    p_all.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "predict" and not args.text and not args.input_file:
        parser.error("predict requires --text or --input-file")

    args.func(args)


if __name__ == "__main__":
    main()
