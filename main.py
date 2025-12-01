import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = "v1beta"
RETRY_DELAY = 1

# 全局状态变量
current_key_index = -1
key_rotation_lock = asyncio.Lock()
http_client: httpx.AsyncClient = None
DEFAULT_SAFETY_SETTINGS = []

try:
    base_path = Path(__file__).parent

    # 加载 config.yaml
    config_path = base_path / "config.yaml"
    if not config_path.exists():
        logger.warning("config.yaml 未找到，将使用默认空配置 (服务可能不可用)。")
        config = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # 加载 safety_settings.json (用于解除 Google 默认的内容审查)
    safety_path = base_path / "safety_settings.json"
    if safety_path.exists():
        try:
            with open(safety_path, "r", encoding="utf-8") as f:
                safety_data = json.load(f)
                DEFAULT_SAFETY_SETTINGS = safety_data.get("default_safety_settings", [])
            logger.info(f"已加载安全设置，包含 {len(DEFAULT_SAFETY_SETTINGS)} 条规则。")
        except Exception as e:
            logger.error(f"安全设置加载失败: {e}")
    else:
        logger.warning("safety_settings.json 未找到，将使用 Google 默认严格审查。")

    # 提取配置参数
    CUSTOM_API_KEYS = config.get("custom_api_keys", [])
    NATIVE_API_KEYS = config.get("native_api_keys", [])
    SERVER_HOST = config.get("host", "0.0.0.0")
    SERVER_PORT = int(config.get("port", 8003))

    REQUEST_MODE = config.get("request_mode", "polling").lower()
    CONCURRENT_BATCH_SIZE = int(config.get("concurrent_batch_size", 3))
    TIMEOUT_SECONDS = 120.0 # 最大响应时间

    # 参数校验
    if REQUEST_MODE not in ["polling", "concurrent"]:
        logger.warning(f"无效的 request_mode '{REQUEST_MODE}'，回退到 polling 模式。")
        REQUEST_MODE = "polling"

    if not isinstance(CUSTOM_API_KEYS, list) or not isinstance(NATIVE_API_KEYS, list):
        raise TypeError("配置错误: keys 必须是列表格式。")

except Exception as e:
    logger.critical(f"服务器初始化失败: {e}")
    exit(1)


def mask_key(key: str, visible_chars=4) -> str:
    if not isinstance(key, str) or len(key) <= visible_chars * 2:
        return "********"
    return f"{key[:visible_chars]}...{key[-visible_chars:]}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 创建全局 HTTP 连接池
    global http_client
    http_client = httpx.AsyncClient(
        timeout=TIMEOUT_SECONDS,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
        follow_redirects=True,
    )
    logger.info(
        f"服务已启动 | 地址: {SERVER_HOST}:{SERVER_PORT} | 模式: {REQUEST_MODE.upper()}"
    )
    if DEFAULT_SAFETY_SETTINGS:
        logger.info("安全过滤器: 已注入")
    else:
        logger.info("安全过滤器: 未启用")

    yield

    # 关闭时：清理资源
    if http_client:
        await http_client.aclose()
    logger.info("服务已关闭，HTTP 连接池已释放。")


app = FastAPI(title="Gemini API Proxy", version="3.9.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _execute_request(
    method: str,
    url: str,
    payload: Dict | None,
    headers: Dict,
    params: Dict,
    is_stream: bool,
) -> httpx.Response:
    """使用全局连接池执行请求的基础函数"""
    req = http_client.build_request(
        method.upper(), url, json=payload, headers=headers, params=params
    )
    response = await http_client.send(req, stream=is_stream)
    response.raise_for_status()
    return response


# 轮询 (Polling)
async def make_request_polling(
    method: str,
    url: str,
    payload: Dict | None,
    is_stream: bool,
    mode: str,
    request: Request,
) -> httpx.Response:
    global current_key_index
    last_exception = None

    async with key_rotation_lock:
        start_index = (current_key_index + 1) % len(NATIVE_API_KEYS)

    for i in range(len(NATIVE_API_KEYS)):
        if await request.is_disconnected():
            raise asyncio.CancelledError("Client disconnected")

        key_idx_to_try = (start_index + i) % len(NATIVE_API_KEYS)
        current_key = NATIVE_API_KEYS[key_idx_to_try]

        headers, params = {}, {}
        if mode == "openai":
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
            }
        else:
            params = {"key": current_key}
            if is_stream:
                params["alt"] = "sse"

        try:
            response = await _execute_request(
                method, url, payload, headers, params, is_stream
            )

            async with key_rotation_lock:
                current_key_index = key_idx_to_try

            logger.info(f"-> [轮询] 密钥 {mask_key(current_key)} 成功")
            return response

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # 429=限速, 5xx=Google服务端错误 -> 此时才换 Key 重试
            if status == 429 or status >= 500:
                logger.warning(
                    f"-> [轮询] 密钥 {mask_key(current_key)} 失败 ({status})，尝试下一个..."
                )
                last_exception = e
                await asyncio.sleep(RETRY_DELAY)
            else:
                # 400, 401 等错误通常是请求本身有问题，换 Key 没用，直接抛出
                raise e
        except Exception as e:
            logger.error(f"-> [轮询] 网络错误: {e}")
            last_exception = e
            await asyncio.sleep(RETRY_DELAY)

    raise last_exception or HTTPException(500, "所有密钥池资源均已耗尽。")


# 并发 (Concurrent)
async def make_request_concurrent(
    method: str,
    url: str,
    payload: Dict | None,
    is_stream: bool,
    mode: str,
    request: Request,
) -> httpx.Response:
    global current_key_index

    async with key_rotation_lock:
        start_index = (current_key_index + 1) % len(NATIVE_API_KEYS)

    # 构造按顺序排列的 Key 列表
    ordered_keys = [
        NATIVE_API_KEYS[(start_index + i) % len(NATIVE_API_KEYS)]
        for i in range(len(NATIVE_API_KEYS))
    ]
    last_exception = None

    async def try_single_key(key: str):
        headers, params = {}, {}
        if mode == "openai":
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
        else:
            params = {"key": key}
            if is_stream:
                params["alt"] = "sse"
        return await _execute_request(method, url, payload, headers, params, is_stream)

    # 分批处理
    for i in range(0, len(ordered_keys), CONCURRENT_BATCH_SIZE):
        if await request.is_disconnected():
            raise asyncio.CancelledError("Client disconnected")

        batch_keys = ordered_keys[i : i + CONCURRENT_BATCH_SIZE]
        logger.info(f"->批次发起 {len(batch_keys)} 个请求...")

        tasks = [asyncio.create_task(try_single_key(key)) for key in batch_keys]
        pending = set(tasks)

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    response = task.result()
                    # 成功！取消其他任务
                    for p in pending:
                        p.cancel()
                    logger.info("->请求成功，已取消同批次其他任务。")
                    return response
                except Exception as e:
                    last_exception = e

        logger.warning(f"->批次全部失败: {last_exception}")

    if last_exception:
        raise last_exception
    raise HTTPException(500, "所有并发尝试均失败。")


async def make_request_dispatch(
    method: str,
    url: str,
    payload: Dict | None,
    is_stream: bool,
    mode: str,
    request: Request,
) -> httpx.Response:
    if not NATIVE_API_KEYS:
        raise HTTPException(500, "服务端未配置 native_api_keys")

    if REQUEST_MODE == "concurrent":
        return await make_request_concurrent(
            method, url, payload, is_stream, mode, request
        )
    return await make_request_polling(method, url, payload, is_stream, mode, request)


def try_parse_json(content: bytes) -> Any:
    """尝试解析 JSON，如果失败返回包含原始文本的错误对象"""
    try:
        return json.loads(content)
    except:
        return {
            "error": {
                "message": content.decode("utf-8", errors="ignore"),
                "type": "upstream_raw_error",
            }
        }


def inject_safety_settings(payload: Dict):
    """注入安全设置到请求体"""
    if (
        payload is not None
        and DEFAULT_SAFETY_SETTINGS
        and "safetySettings" not in payload
    ):
        payload["safetySettings"] = DEFAULT_SAFETY_SETTINGS


# OpenAI 兼容接口代理
async def unified_proxy_logic(request: Request, upstream_url: str):
    response = None
    try:
        # 鉴权
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer Token")
        client_key = auth_header.split("Bearer ")[1].strip()

        # 解析请求体
        method = request.method
        payload = await request.json() if method == "POST" else {}
        is_stream = payload.get("stream", False)

        # 注入安全设置
        inject_safety_settings(payload)

        # 转发请求
        if client_key in CUSTOM_API_KEYS:
            # 使用服务器密钥池
            response = await make_request_dispatch(
                method, upstream_url, payload, is_stream, "openai", request
            )
        else:
            # 直连模式 (使用客户端提供的 Key)
            headers = {
                "Authorization": f"Bearer {client_key}",
                "Content-Type": "application/json",
            }
            response = await _execute_request(
                method, upstream_url, payload, headers, {}, is_stream
            )

        # 返回响应
        if is_stream:

            async def stream_generator(resp: httpx.Response):
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                except Exception as e:
                    logger.error(f"Stream Error: {e}")
                    yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode()
                finally:
                    await resp.aclose()

            return StreamingResponse(
                stream_generator(response), media_type="text/event-stream"
            )
        else:
            try:
                content = await response.aread()
                try:
                    data = json.loads(content)
                    return JSONResponse(content=data, status_code=response.status_code)
                except json.JSONDecodeError:
                    return Response(content=content, status_code=response.status_code)
            finally:
                await response.aclose()

    except httpx.HTTPStatusError as e:
        await e.response.aread()
        return JSONResponse(
            status_code=e.response.status_code,
            content=try_parse_json(e.response.content),
        )
    except Exception as e:
        logger.error(f"Proxy Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


# Google 原生接口代理
async def native_proxy_logic(request: Request, url_path: str):
    response = None
    try:
        # 获取 Key
        client_key = request.headers.get("x-goog-api-key") or request.query_params.get(
            "key"
        )
        if not client_key:
            raise HTTPException(401, "Missing x-goog-api-key")

        target_url = f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/{url_path}"
        method = request.method
        payload = await request.json() if method == "POST" else None
        is_stream = (
            "stream" in url_path.lower() or request.query_params.get("alt") == "sse"
        )

        # 注入安全设置
        inject_safety_settings(payload)

        # 转发
        if client_key in CUSTOM_API_KEYS:
            response = await make_request_dispatch(
                method, target_url, payload, is_stream, "native", request
            )
        else:
            params = dict(request.query_params)
            params["key"] = client_key
            response = await _execute_request(
                method, target_url, payload, {}, params, is_stream
            )

        # 返回
        if is_stream:

            async def native_generator(resp):
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode()
                finally:
                    await resp.aclose()

            return StreamingResponse(
                native_generator(response),
                media_type="text/event-stream",
                status_code=response.status_code,
            )
        else:
            try:
                content = await response.aread()
                try:
                    data = json.loads(content)
                    return JSONResponse(data, status_code=response.status_code)
                except json.JSONDecodeError:
                    return Response(content=content, status_code=response.status_code)
            finally:
                await response.aclose()

    except httpx.HTTPStatusError as e:
        await e.response.aread()
        return JSONResponse(
            status_code=e.response.status_code,
            content=try_parse_json(e.response.content),
        )
    except Exception as e:
        logger.error(f"Native Proxy Error: {e}")
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


@app.get("/", response_class=HTMLResponse)
async def root():
    status = "Active" if DEFAULT_SAFETY_SETTINGS else "Inactive (Default Strict)"
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        content = content.replace("{{version}}", "v4.0.0")
        content = content.replace("{{mode}}", REQUEST_MODE.capitalize())
        content = content.replace("{{safety}}", status)
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(
            f"<h1>Gemini Proxy v4.0.0</h1><p>Mode: {REQUEST_MODE}</p><p>Safety Filter: {status}</p>"
        )


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    return await unified_proxy_logic(
        request, f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/openai/chat/completions"
    )


@app.api_route("/v1/images/generations", methods=["POST"])
async def image_generations(request: Request):
    return await unified_proxy_logic(
        request, f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/openai/images/generations"
    )


@app.api_route("/v1/embeddings", methods=["POST"])
async def embeddings(request: Request):
    return await unified_proxy_logic(
        request, f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/openai/embeddings"
    )


@app.get("/v1/models")
async def list_models(request: Request):
    return await unified_proxy_logic(
        request, f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/openai/models"
    )


@app.get("/v1/models/{model_id:path}")
async def get_model(request: Request, model_id: str):
    return await unified_proxy_logic(
        request, f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/openai/models/{model_id}"
    )


@app.get("/v1beta/models")
async def list_native_models(request: Request):
    return await native_proxy_logic(request, "models")


@app.post("/v1beta/models/{model_and_action:path}")
async def handle_native_requests(model_and_action: str, request: Request):
    return await native_proxy_logic(request, f"models/{model_and_action}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

