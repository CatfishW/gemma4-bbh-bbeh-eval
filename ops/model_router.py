#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
import httpx
from starlette.responses import JSONResponse, StreamingResponse

from ops.model_router_core import parse_backends, select_model

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


MODEL_BACKENDS = parse_backends(os.getenv("MODEL_BACKENDS"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "SubTokenLLM")
if DEFAULT_MODEL not in MODEL_BACKENDS:
    raise RuntimeError("DEFAULT_MODEL must exist in MODEL_BACKENDS")


def request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
    }


def response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient(
        timeout=None,
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="Gemma 4 model router", lifespan=lifespan)


@app.get("/healthz")
async def health() -> dict:
    return {
        "status": "ok",
        "models": sorted(MODEL_BACKENDS),
        "request_body_mutation": False,
        "system_prompt_injection": False,
    }


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    async def check(model: str, base_url: str) -> dict:
        try:
            response = await request.app.state.client.get(f"{base_url}/models")
            response.raise_for_status()
            upstream = response.json()
            upstream_row = next(iter(upstream.get("data", [])), {})
            return {
                "id": model,
                "object": "model",
                "created": upstream_row.get("created", 0),
                "owned_by": upstream_row.get("owned_by", "sglang"),
                "root": model,
                "parent": None,
                "max_model_len": upstream_row.get("max_model_len"),
            }
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return {"id": model, "object": "model", "available": False, "error": str(exc)}

    rows = await asyncio.gather(
        *(check(model, base_url) for model, base_url in MODEL_BACKENDS.items())
    )
    return JSONResponse({"object": "list", "data": rows})


@app.api_route(
    "/v1/{upstream_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(request: Request, upstream_path: str) -> StreamingResponse:
    body = await request.body()
    try:
        model = select_model(body, MODEL_BACKENDS, DEFAULT_MODEL)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    base_url = MODEL_BACKENDS[model]
    query = f"?{request.url.query}" if request.url.query else ""
    upstream_url = f"{base_url}/{upstream_path}{query}"
    upstream_request = request.app.state.client.build_request(
        request.method,
        upstream_url,
        headers=request_headers(request),
        content=body,
    )
    try:
        upstream_response = await request.app.state.client.send(
            upstream_request,
            stream=True,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream request failed: {exc}") from exc

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers(upstream_response),
    )
