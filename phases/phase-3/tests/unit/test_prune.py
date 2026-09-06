import numpy as np
import pandas as pd

from src.models.prune import prune_pipeline, rank_features
from src.models.train import build_pipeline

URGENT = "acute myocardial infarction chest pain troponin elevation coronary occlusion"
ATTENTION = "colon adenocarcinoma tumor resection lymph node staging chemotherapy"
NORMAL = "routine screening stable metabolic parameters without acute inflammatory findings"


def _fitted_pipeline():
    rows = []
    for i in range(15):
        rows.append({"text": f"{URGENT} case {i}", "urgency": "urgente"})
        rows.append({"text": f"{ATTENTION} case {i}", "urgency": "atencao"})
        rows.append({"text": f"{NORMAL} case {i}", "urgency": "normal"})
    df = pd.DataFrame(rows)

    pipeline = build_pipeline(min_df=1)
    pipeline.fit(df["text"], df["urgency"])
    return pipeline, df


def test_rank_features_orders_by_descending_weight_magnitude():
    pipeline, _ = _fitted_pipeline()

    ranking = rank_features(pipeline)
    importance = np.abs(pipeline.named_steps["classifier"].coef_).max(axis=0)

    assert len(ranking) == importance.shape[0]
    assert importance[ranking[0]] >= importance[ranking[-1]]


def test_prune_pipeline_shrinks_the_vocabulary_to_the_requested_size():
    pipeline, df = _fitted_pipeline()
    original_size = len(pipeline.named_steps["tfidf"].vocabulary_)

    pruned = prune_pipeline(pipeline, df["text"], df["urgency"], keep_features=10)

    assert len(pruned.named_steps["tfidf"].vocabulary_) == 10
    assert original_size > 10


def test_prune_pipeline_keeps_the_model_usable():
    pipeline, df = _fitted_pipeline()

    pruned = prune_pipeline(pipeline, df["text"], df["urgency"], keep_features=20)

    assert pruned.predict([URGENT])[0] == "urgente"
    assert pruned.predict_proba([NORMAL]).shape == (1, 3)


def test_prune_pipeline_is_a_noop_when_the_target_exceeds_the_vocabulary():
    pipeline, df = _fitted_pipeline()

    pruned = prune_pipeline(pipeline, df["text"], df["urgency"], keep_features=10**6)

    assert pruned is pipeline
