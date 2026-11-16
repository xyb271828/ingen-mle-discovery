import unittest

import httpx
from fastapi.testclient import TestClient

from gateway import create_app


class GatewayTests(unittest.TestCase):
    def test_organic_survives_unavailable_ads(self):
        def respond(request):
            if request.url.path == "/auction":
                self.assertIn(b'"topic":"fractions"', request.content)
                self.assertNotIn(b"user_id", request.content)
                return httpx.Response(503, json={"error":"unavailable"})
            return httpx.Response(200, json={"results":[{"id":"lesson-1"}],"mode":"lambdamart"})

        with TestClient(create_app(httpx.MockTransport(respond))) as client:
            result = client.post("/lessons", json={"query":"equivalent fractions","grade":5}).json()
        self.assertEqual(result["organic"], [{"id":"lesson-1"}])
        self.assertIsNone(result["sponsored"])
        self.assertEqual(result["sponsored_status"], "ads_unavailable")

    def test_ambiguous_query_does_not_call_ads(self):
        def respond(request):
            self.assertEqual(request.url.path, "/search")
            return httpx.Response(200, json={"results":[{"id":"lesson-2"}],"mode":"lambdamart"})
        with TestClient(create_app(httpx.MockTransport(respond))) as client:
            result = client.post("/lessons", json={"query":"fractions and geometry","grade":5}).json()
        self.assertEqual(result["sponsored_status"], "no_context_match")
