"""Serve Week 2 frozen retrieval, rank features, and MMR on synthetic activities."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter_ns

import faiss
import lightgbm
import numpy as np
from fastapi import FastAPI, Query

from recsys.benchmark import baseline_models, positive_matrix
from recsys.data import synthetic
from recsys.production import build_feed, recency_features, recent_candidates, run


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    run(directory)
    seed = 42
    data = synthetic(seed)
    matrix, counts, recent = positive_matrix(data)
    als = baseline_models(matrix, counts, recent, seed)["als"]
    activity_counts, affinity, categories = recency_features(data)
    training = data.events[data.events[:, 2] <= data.train_end]
    data.events = training
    histories = np.minimum(np.bincount(training[training[:, 4] == 1, 0], minlength=data.users), 16)
    context = [(i, data.train_end + 1, h) for i in range(data.users) for h in range(4)]
    exclusions = np.zeros((data.users, 4, data.catalog.size), dtype=bool)
    for (user, _, hour), blocked in zip(context, recent_candidates(data, context), strict=True):
        exclusions[user, hour, list(blocked)] = True
    index = faiss.read_index(str(directory / "activities.faiss"))
    embeddings = index.reconstruct_n(0, index.ntotal)
    np.savez(directory / "feed.npz", als=als(np.arange(data.users)), counts=activity_counts,
             affinity=affinity, categories=categories, history=histories, excludes=exclusions,
             embeddings=embeddings, titles=data.catalog.titles, labels=data.catalog.categories)


def create_app():
    directory = Path(os.getenv("FARI_ARTIFACT_DIR", "/artifacts"))
    index = faiss.read_index(str(directory / "activities.faiss"))
    with np.load(directory / "users.npz", allow_pickle=False) as user:
        vectors = dict(zip(user["user_ids"].tolist(), user["vectors"], strict=True))
    with np.load(directory / "feed.npz", allow_pickle=False) as bundle:
        state = dict(bundle.items())
    popularity = json.loads((directory / "popularity.json").read_text())
    ranker = lightgbm.Booster(model_file=str(directory / "ranker.txt"))
    mmr_weight = json.loads((directory / "ranking.json").read_text())["mmr_weight"]
    embeddings = state["embeddings"]
    similarity = embeddings @ embeddings.T
    app = FastAPI(title="Synthetic activity feed")

    @app.get("/health")
    def health():
        return {"ready": True}

    @app.get("/feed")
    def feed(user_id: int = Query(ge=0), k: int = Query(10, ge=1, le=30),
             hour_bucket: int = Query(1, ge=0, le=3)):
        start = perf_counter_ns()
        timing = {}
        if user_id not in vectors:
            ids, source = popularity[:k], "popularity"
        else:
            faiss.omp_set_num_threads(1)
            query = vectors[user_id][hour_bucket]
            _, candidates = index.search(query.reshape(1,-1), 200)
            ids = candidates[0, candidates[0] >= 0]
            extracted = perf_counter_ns()
            affinity = state["affinity"][user_id]
            features = np.column_stack((
                embeddings[ids] @ query, state["als"][user_id, ids],
                affinity[state["categories"][ids]] / max(int(affinity.sum()), 1),
                state["counts"][0, ids], state["counts"][1, ids],
                np.full(len(ids), hour_bucket), np.full(len(ids), state["history"][user_id]),
            ))
            scores = ranker.predict(features, num_threads=1)
            scored = perf_counter_ns()
            excluded = set(np.flatnonzero(state["excludes"][user_id, hour_bucket]))
            ids = build_feed(ids, scores, similarity, mmr_weight, excluded, k=k)
            timing = {"retrieval": (extracted-start)/1e6, "rank": (scored-extracted)/1e6,
                      "mmr": (perf_counter_ns()-scored)/1e6}
            source = "hnsw_lgbm_mmr"
        items = [{"id": int(i), "title": str(state["titles"][i]), "category": str(state["labels"][i])}
                 for i in ids]
        timing["feed"] = (perf_counter_ns()-start)/1e6
        return {"source": source, "items": items, "timing_ms": timing}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    prepare(parser.parse_args().directory)
