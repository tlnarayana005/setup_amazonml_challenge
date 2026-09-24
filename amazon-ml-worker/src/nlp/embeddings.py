"""
Amazon ML Worker — Text Embeddings.

Optional text embedding extraction. Only loads models when explicitly configured.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.etl.cache import CacheManager
from src.utils.logging import get_logger

log = get_logger(__name__)


def extract_text_embeddings(
    df: pd.DataFrame,
    id_column: str = "id",
    text_column: str = "text",
    model_name: str = "",
    embedding_dim: int = 384,
    batch_size: int = 32,
    cache_dir: str = "cache",
    resume: bool = True,
) -> pd.DataFrame:
    """
    Extract text embeddings.

    If no model is specified, returns dummy zero embeddings (placeholder).
    Real model loading requires sentence-transformers or transformers.

    Returns DataFrame with columns [id, text_emb_0, ..., text_emb_{dim-1}].
    """
    cache = CacheManager(cache_dir, cache_key=f"text_emb_{model_name or 'dummy'}")

    if resume:
        df = cache.filter_uncached(df, id_column=id_column)

    if len(df) == 0:
        cached = cache.get_cached_data()
        if cached is not None:
            return cached
        return _empty_embedding_df(id_column, embedding_dim)

    if not model_name:
        log.info("No text embedding model specified. Generating placeholders.")
        return _placeholder_embeddings(df, id_column, embedding_dim, cache)

    try:
        return _model_embeddings(
            df, id_column, text_column,
            model_name, embedding_dim, batch_size, cache,
        )
    except ImportError as e:
        log.warning("Cannot load text embedding model (missing dep: %s). Using placeholders.", e)
        return _placeholder_embeddings(df, id_column, embedding_dim, cache)


def _placeholder_embeddings(df, id_column, embedding_dim, cache):
    ids = df[id_column].values
    emb_cols = [f"text_emb_{i}" for i in range(embedding_dim)]
    data = {id_column: ids}
    for col in emb_cols:
        data[col] = np.zeros(len(ids), dtype=np.float32)
    result = pd.DataFrame(data)
    cache.save_to_cache(result, append=True)
    return result


def _model_embeddings(df, id_column, text_column, model_name, embedding_dim, batch_size, cache):
    """Try sentence-transformers, fall back to HF transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        texts = df[text_column].fillna("").astype(str).tolist()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    except ImportError:
        # Fallback: HF transformers
        import torch
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        texts = df[text_column].fillna("").astype(str).tolist()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
            # Mean pooling
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embs.append(emb)
        embeddings = np.concatenate(all_embs, axis=0)

    # Pad/truncate
    if embeddings.shape[1] < embedding_dim:
        pad = np.zeros((embeddings.shape[0], embedding_dim - embeddings.shape[1]))
        embeddings = np.concatenate([embeddings, pad], axis=1)
    else:
        embeddings = embeddings[:, :embedding_dim]

    emb_cols = [f"text_emb_{i}" for i in range(embedding_dim)]
    result = pd.DataFrame(embeddings, columns=emb_cols)
    result.insert(0, id_column, df[id_column].values)
    cache.save_to_cache(result, append=True)
    return result


def _empty_embedding_df(id_column, embedding_dim):
    cols = [id_column] + [f"text_emb_{i}" for i in range(embedding_dim)]
    return pd.DataFrame(columns=cols)
