"""Route lessons to organic search and contextual ads and activities to Fari."""
import asyncio
import os
import re
from contextlib import asynccontextmanager
from time import perf_counter_ns
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


class LessonQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(5, ge=1, le=20)
    grade: int = Field(ge=1, le=12)


def match_topic(query):
    patterns = {"fractions": r"\bfractions?\b", "geometry": r"\b(geometry|angles?|shapes?)\b",
                "reading": r"\breading\b"}
    found = [topic for topic, pattern in patterns.items() if re.search(pattern, query, re.I)]
    return found[0] if len(found) == 1 else None


def create_app(transport=None):
    search_url = os.getenv("SEARCH_URL", "http://search:8000")
    ads_url = os.getenv("ADS_URL", "http://ads:8000")
    fari_url = os.getenv("FARI_URL", "http://fari:8000")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, timeout=2) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Discovery systems", lifespan=lifespan)

    async def upstream(method, url, **options):
        start = perf_counter_ns()
        reply = await app.state.client.request(method, url, **options)
        reply.raise_for_status()
        return reply.json(), (perf_counter_ns()-start)/1e6

    async def sponsored(topic, grade):
        if not topic:
            return None, None, "no_context_match"
        try:
            result, elapsed = await asyncio.wait_for(
                upstream("POST", ads_url + "/auction",
                         json={"topic": topic, "grade": grade, "min_relevance": .5}),
                timeout=.1)
            slot = result.get("sponsored")
            return slot, elapsed, "shown" if slot else "no_eligible_sponsor"
        except (TimeoutError, httpx.HTTPError, ValueError):
            return None, None, "ads_unavailable"

    @app.get("/health")
    def health():
        return {"ready": True}

    @app.post("/lessons")
    async def lessons(request: LessonQuery):
        started = perf_counter_ns()
        if not request.query.strip():
            raise HTTPException(422, "Query must include visible text")
        try:
            (organic, search_ms), (slot, ads_ms, reason) = await asyncio.gather(
                upstream("POST", search_url + "/search",
                         json={"query": request.query, "k": request.k, "mode": "lambdamart", "rerank_depth": 30}),
                sponsored(match_topic(request.query), request.grade),
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, "Search unavailable") from exc
        return {"request_id": str(uuid4()), "organic": organic["results"], "sponsored": slot,
                "sponsored_status": reason, "search_mode": organic["mode"],
                "timing_ms": {"search_http": search_ms, "ads_http": ads_ms,
                              "gateway": (perf_counter_ns()-started)/1e6}}

    @app.get("/activities")
    async def activities(user_id: int = Query(ge=0), k: int = Query(10, ge=1, le=30),
                         hour_bucket: int = Query(1, ge=0, le=3)):
        start = perf_counter_ns()
        try:
            result, elapsed = await upstream("GET", fari_url + "/feed",
                                             params={"user_id": user_id, "k": k, "hour_bucket": hour_bucket})
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, "Recommendations unavailable") from exc
        return {**result, "request_id": str(uuid4()), "timing_ms": {
            **result["timing_ms"], "fari_http": elapsed, "gateway": (perf_counter_ns()-start)/1e6}}

    return app
