import time

import structlog
from fastapi import Request

logger = structlog.get_logger()


async def structlog_middleware(request: Request, call_next):
    structlog.contextvars.clear_contextvars()

    structlog.contextvars.bind_contextvars(
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )

    logger.info("request_started")

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        process_time = time.perf_counter() - start_time
        logger.exception(
            "request_failed",
            duration_ms=round(process_time * 1000, 2),
        )
        raise

    process_time = time.perf_counter() - start_time

    logger.info(
        "request_finished",
        status_code=response.status_code,
        duration_ms=round(process_time * 1000, 2),
    )

    return response
