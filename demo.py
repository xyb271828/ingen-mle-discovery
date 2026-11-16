"""Exercise the public gateway with a matched sponsor, cold start and known user."""
import json
from urllib.request import Request, urlopen


ROOT = "http://127.0.0.1:8080"


def get(path, body=None):
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(ROOT+path, data=data, headers=headers), timeout=10) as response:
        return json.load(response)


def main():
    organic = get("/lessons", {"query":"Grade 5 beginner practice on equivalent fractions",
                                "grade":5, "k":3})
    unmatched = get("/lessons", {"query":"Grade 5 writing practice", "grade":5, "k":3})
    known = get("/activities?user_id=42&k=5&hour_bucket=1")
    cold = get("/activities?user_id=999999&k=5&hour_bucket=1")
    assert organic["organic"] and organic["sponsored"]["label"] == "Sponsored"
    assert unmatched["organic"] and unmatched["sponsored"] is None
    assert known["source"] == "hnsw_lgbm_mmr" and len(known["items"]) == 5
    assert cold["source"] == "popularity" and len(cold["items"]) == 5
    print(json.dumps({"lessons":organic, "matched_activity_feed":known, "cold_start":cold},
                     indent=2))


if __name__ == "__main__":
    main()
