"""
Amazon ML Worker — Image Embeddings.

Optional image embedding extraction using pretrained models.
Only loaded when explicitly enabled via configuration.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cv.image_utils import load_image
from src.etl.cache import CacheManager
from src.utils.logging import get_logger

log = get_logger(__name__)


def extract_image_embeddings(
    df: pd.DataFrame,
    id_column: str = "id",
    image_column: str = "image",
    image_dir: str = "data/raw/images",
    model_name: str = "",
    embedding_dim: int = 384,
    batch_size: int = 32,
    cache_dir: str = "cache",
    resume: bool = True,
) -> pd.DataFrame:
    """
    Extract image embeddings.

    If no model is specified, returns dummy zero embeddings (placeholder).
    Real model loading requires torchvision/timm which are optional.

    Returns DataFrame with columns [id, emb_0, emb_1, ..., emb_{dim-1}].
    """
    cache = CacheManager(cache_dir, cache_key=f"img_emb_{model_name or 'dummy'}")

    if resume:
        df = cache.filter_uncached(df, id_column=id_column)

    if len(df) == 0:
        cached = cache.get_cached_data()
        if cached is not None:
            return cached
        return _empty_embedding_df(id_column, embedding_dim)

    if not model_name:
        log.info("No embedding model specified. Generating placeholder embeddings.")
        return _placeholder_embeddings(df, id_column, embedding_dim, cache)

    # Try loading a real model
    try:
        return _model_embeddings(
            df, id_column, image_column, image_dir,
            model_name, embedding_dim, batch_size, cache,
        )
    except ImportError as e:
        log.warning("Cannot load embedding model (missing dependency: %s). Using placeholders.", e)
        return _placeholder_embeddings(df, id_column, embedding_dim, cache)


def _placeholder_embeddings(
    df: pd.DataFrame,
    id_column: str,
    embedding_dim: int,
    cache: CacheManager,
) -> pd.DataFrame:
    """Generate zero embeddings as placeholders."""
    ids = df[id_column].values
    emb_cols = [f"img_emb_{i}" for i in range(embedding_dim)]
    data = {id_column: ids}
    for col in emb_cols:
        data[col] = np.zeros(len(ids), dtype=np.float32)
    result = pd.DataFrame(data)
    cache.save_to_cache(result, append=True)
    return result


def _model_embeddings(
    df, id_column, image_column, image_dir,
    model_name, embedding_dim, batch_size, cache,
):
    """Real model embedding extraction (requires torch + timm/torchvision)."""
    import torch

    # Try timm first
    try:
        import timm
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    except Exception:
        log.warning("Could not load model '%s' from timm.", model_name)
        return _placeholder_embeddings(df, id_column, embedding_dim, cache)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_embeddings = []
    all_ids = []
    image_dir_path = Path(image_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Image embeddings"):
        row_id = str(row[id_column])
        img_ref = str(row.get(image_column, ""))
        img_path = image_dir_path / img_ref

        img = load_image(str(img_path), target_size=(224, 224))
        if img is None:
            all_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            all_ids.append(row_id)
            continue

        # Normalize and convert to tensor
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten()

        # Pad or truncate to expected dim
        if len(emb) < embedding_dim:
            emb = np.pad(emb, (0, embedding_dim - len(emb)))
        else:
            emb = emb[:embedding_dim]

        all_embeddings.append(emb)
        all_ids.append(row_id)

    # Build DataFrame
    emb_cols = [f"img_emb_{i}" for i in range(embedding_dim)]
    emb_array = np.stack(all_embeddings)
    result = pd.DataFrame(emb_array, columns=emb_cols)
    result.insert(0, id_column, all_ids)

    cache.save_to_cache(result, append=True)
    return result


def _empty_embedding_df(id_column: str, embedding_dim: int) -> pd.DataFrame:
    cols = [id_column] + [f"img_emb_{i}" for i in range(embedding_dim)]
    return pd.DataFrame(columns=cols)
