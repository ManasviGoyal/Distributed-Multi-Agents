import pytest
import asyncio
import importlib.util
import sys
import os
from unittest import mock
from fastapi.testclient import TestClient
from httpx import AsyncClient, Response, RequestError

# Load load_balancer.py from src
module_name = "src.load_balancer"
file_path = os.path.join(os.path.dirname(__file__), "../src/load_balancer.py")
spec = importlib.util.spec_from_file_location(module_name, file_path)
load_balancer = importlib.util.module_from_spec(spec)
sys.modules[module_name] = load_balancer
spec.loader.exec_module(load_balancer)

client = TestClient(load_balancer.app)

@pytest.mark.asyncio
async def test_proxy_round_robin(monkeypatch):
    """Test non-sticky routing (round-robin behavior)"""
    request_count = []

    async def mock_request(*args, **kwargs):
        request_count.append(kwargs['url'])
        return Response(200, content=b"OK", headers={"content-type": "text/plain"})

    monkeypatch.setattr(AsyncClient, "request", mock_request)

    # Fire 3 different non-sticky requests
    for _ in range(3):
        response = client.get("/process_query")
        assert response.status_code == 200
        assert response.text == "OK"

    assert len(set(request_count)) == 3  # Should hit 3 unique backends (round robin)

@pytest.mark.asyncio
async def test_proxy_sticky_routing(monkeypatch):
    """Test that sticky routing is hash-based and varies by attempt"""
    urls = []

    async def mock_request(*args, **kwargs):
        urls.append(kwargs["url"])
        return Response(200, content=b"STICKY", headers={"content-type": "text/plain"})

    monkeypatch.setattr(AsyncClient, "request", mock_request)

    job_id = "abc123"
    path = f"/job_status/{job_id}"

    for _ in range(3):
        response = client.get(path)
        assert response.status_code == 200
        assert response.text == "STICKY"

    # Check all requests target the correct endpoint
    assert all(path in url for url in urls)

    # Expect 3 URLs because hash(job_id + attempt) varies
    assert len(urls) == 3
    assert len(set(urls)) == 3  # Different backends used due to retry logic

@pytest.mark.asyncio
async def test_proxy_backend_failure_and_retry(monkeypatch):
    """Test fallback to next backend on failure"""
    call_log = []

    async def failing_then_successful_request(*args, **kwargs):
        call_log.append(kwargs["url"])
        # Fail first two times
        if len(call_log) < 3:
            raise RequestError("Backend down")
        return Response(200, content=b"RECOVERED", headers={"content-type": "text/plain"})

    monkeypatch.setattr(AsyncClient, "request", failing_then_successful_request)

    response = client.get("/process_query")
    assert response.status_code == 200
    assert response.text == "RECOVERED"
    assert len(call_log) == 3  # Should have retried 3 times max

@pytest.mark.asyncio
async def test_proxy_all_backends_fail(monkeypatch):
    """Test that 503 is returned when all backends fail"""
    monkeypatch.setattr(
        AsyncClient, "request", mock.AsyncMock(side_effect=RequestError("fail"))
    )

    response = client.get("/process_query")
    assert response.status_code == 503
    assert "All backends failed" in response.text
