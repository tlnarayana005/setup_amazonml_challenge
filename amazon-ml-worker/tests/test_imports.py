"""Test all critical imports."""
import pytest


def test_import_config():
    from src.config.loader import WorkerConfig, load_config

def test_import_hardware():
    from src.hardware.detect import detect_hardware

def test_import_data():
    from src.data.inspect import inspect_dataframe
    from src.data.validate import validate_dataframe
    from src.data.split import create_split
    from src.data.shard import get_shard
    from src.data.merge import merge_worker_outputs

def test_import_etl():
    from src.etl.pipeline import run_etl
    from src.etl.preprocess import preprocess_dataframe
    from src.etl.cache import CacheManager

def test_import_ocr():
    from src.ocr.base import OCRBackend, run_ocr_batch
    from src.ocr.tesseract_backend import TesseractBackend
    from src.ocr.paddleocr_backend import PaddleOCRBackend

def test_import_cv():
    from src.cv.image_utils import load_image, validate_image
    from src.cv.embeddings import extract_image_embeddings

def test_import_nlp():
    from src.nlp.text_utils import clean_text, compute_text_stats
    from src.nlp.embeddings import extract_text_embeddings

def test_import_features():
    from src.features.builder import build_features
    from src.features.merger import merge_features

def test_import_models():
    from src.models.baseline import BaselineModel, train_baseline
    from src.models.lightweight import LightweightExperiment
    from src.models.qlora_smoke import check_qlora_dependencies

def test_import_evaluation():
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.threshold import search_threshold

def test_import_inference():
    from src.inference.predict import batch_predict

def test_import_experiments():
    from src.experiments.tracker import ExperimentTracker

def test_import_utils():
    from src.utils.logging import setup_logging, get_logger
    from src.utils.seeds import set_seed
    from src.utils.paths import project_root, ensure_dir
    from src.utils.timing import Timer
