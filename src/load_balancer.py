from fastapi import FastAPI, Request, Response
import httpx
import itertools
import re
import hashlib

app = FastAPI()

# Define backend servers
BACKENDS = ["http://172.31.34.248:8001", "http://172.31.34.248:8002", "http://172.31.34.248:8003"]
backend_cycle = itertools.cycle(BACKENDS)
MAX_RETRIES = 3

def consistent_hash(job_id):
    return int(hashlib.md5(job_id.encode()).hexdigest(), 16)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    """
    Smart proxy:
    - Sticky routing for job_id-specific paths (status/result/image)
    - Round-robin with retry logic for everything else (2-fault tolerant)
    """
    job_id_match = re.search(r"/(?:image|job_status|job_result)/([a-f0-9\-]+)", path)

    for attempt in range(MAX_RETRIES):
        if job_id_match:
            job_id = job_id_match.group(1)
            server_idx = consistent_hash(job_id + str(attempt)) % len(BACKENDS)
            backend = BACKENDS[server_idx]
        elif path == "process_query":
            backend = next(backend_cycle)  # still random for new jobs
        else:
            backend = next(backend_cycle)

        url = f"{backend}/{path}"
        print(f"[Load Balancer] Attempt {attempt + 1}: Routing '{path}' to → {backend}")

        try:
            async with httpx.AsyncClient() as client:
                req_data = await request.body()
                headers = dict(request.headers)

                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=req_data,
                    params=request.query_params
                )

                content_type = resp.headers.get("content-type", "application/octet-stream")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=content_type
                )

        except httpx.RequestError as e:
            print(f"⚠️ Backend {backend} failed with error: {e}")
            continue  # try next backend

    return Response(content="❌ All backends failed.", status_code=503)
