# Discovery Systems

A Docker Compose project that connects three independently runnable ML projects. The lesson path combines organic retrieval and contextual sponsorship. The activity path serves two-tower ANN retrieval, a learned ranker, diversity reranking and a popularity fallback.

| Component | Source | Runtime role |
| --- | --- | --- |
| Lessons | [Week 1 Search](https://github.com/xyb271828/SciFact_Search) | BM25 and MiniLM/FAISS retrieval, reciprocal rank fusion, LambdaMART |
| Activities | [Week 2 Recommender](https://github.com/xyb271828/fari-recsys) | HNSW candidates, LightGBM ranker and MMR |
| Sponsors | [Week 3 Ads](https://github.com/xyb271828/ads-ctr-marketplace) | Calibrated synthetic pCTR plus quality-weighted GSP with a fixed CPC reserve |

## Run the demo

Use Docker Engine with Compose v2. The first boot downloads the public search encoder, builds the images and prepares the synthetic lesson and activity artifacts. It can take several minutes.

```sh
docker compose up --build -d --wait --wait-timeout 900
python3 demo.py
docker compose down
```

The gateway listens on `127.0.0.1:8080`. `POST /lessons` takes a query, grade, and result count. It emits the organic list and a separate labeled sponsored slot when one exists. An unmatched or ambiguous topic returns organic results. The ads call has an independent timeout. `GET /activities` takes a synthetic user ID and hour bucket and returns ranked activities. Unknown IDs fall back to popularity. Responses include request IDs and measured stage timings.

## Data and measurement

Lesson and activity records are generated; sponsor scoring uses topic and grade only. The CTR benchmark in the ads repository is an independent experiment. The activity serving snapshot uses events through the synthetic training cutoff and reuses Week 2's frozen rank model and MMR operating point. Timing fields use per-request wall time measured at each service. Independent Week 1–3 benchmark figures describe their respective workloads and hardware.

The GitHub Actions smoke check exercises sponsored and organic-only search, a personalized activity feed, and cold-start recommendations. Gateway tests also verify organic search when the ads service is unavailable.
