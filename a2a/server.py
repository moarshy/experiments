# server.py
import asyncio, time, json, uuid, hashlib
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.routing import Route
from pydantic import BaseModel
from typing import Dict, Any, AsyncIterable
from jwt import PyJWK, PyJWKClient
from jwcrypto import jwk
import jwt

PORT = 8000

# In-memory
tasks: Dict[str, str] = {}
subscribers: Dict[str, asyncio.Queue] = {}

# -- AUTH SHARED --
class JWTAuth:
    def __init__(self):
        key = jwk.JWK.generate(kty="RSA", size=2048, kid=str(uuid.uuid4()), use="sig")
        self.private_key = PyJWK.from_json(key.export_private())
        self.public_jwk = key.export_public(as_dict=True)

    def expose_jwks(self, _: Request):
        return JSONResponse({"keys": [self.public_jwk]})

    def verify_request(self, token: str, body: dict) -> bool:
        client = PyJWKClient(f"http://localhost:{PORT}/.well-known/jwks.json")
        key = client.get_signing_key_from_jwt(token)
        decoded = jwt.decode(token, key.key, algorithms=["RS256"], options={"require": ["iat", "request_body_sha256"]})
        actual_hash = hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
        if decoded["request_body_sha256"] != actual_hash:
            raise ValueError("Hash mismatch")
        if time.time() - decoded["iat"] > 300:
            raise ValueError("Token expired")
        return True

auth = JWTAuth()

class JSONRPCRequest(BaseModel):
    jsonrpc: str
    id: str
    method: str
    params: Dict[str, Any]

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    result: Any = None
    error: Any = None

def unauthorized(reason="Unauthorized"):
    return JSONResponse({"error": reason}, status_code=401)

# Endpoints
async def send_task(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    body = await request.json()
    try:
        if not auth.verify_request(token, body):
            return unauthorized()
    except Exception as e:
        return unauthorized(str(e))

    rpc = JSONRPCRequest(**body)
    task_id = rpc.params.get("id", str(uuid.uuid4()))
    text = rpc.params.get("text", "")
    tasks[task_id] = f"Processed: {text}"
    return JSONResponse(JSONRPCResponse(id=rpc.id, result={"id": task_id, "message": tasks[task_id]}).model_dump())

async def send_task_subscribe(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    body = await request.json()
    try:
        if not auth.verify_request(token, body):
            return unauthorized()
    except Exception as e:
        return unauthorized(str(e))

    rpc = JSONRPCRequest(**body)
    task_id = rpc.params.get("id", str(uuid.uuid4()))
    text = rpc.params.get("text", "")
    queue = asyncio.Queue()
    subscribers[task_id] = queue

    async def stream_words():
        for word in text.split():
            await asyncio.sleep(0.5)
            await queue.put(word)
        await queue.put("[done]")

    asyncio.create_task(stream_words())

    async def event_gen() -> AsyncIterable[Dict[str, str]]:
        while True:
            word = await queue.get()
            yield {"data": word}
            if word == "[done]":
                break

    return EventSourceResponse(event_gen())

async def push_receiver(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    body = await request.json()
    try:
        if auth.verify_request(token, body):
            print("✅ Push Verified:", body)
            return JSONResponse({"ok": True})
    except Exception as e:
        print("❌ Push Invalid:", str(e))
        return unauthorized(str(e))

# App
app = Starlette(
    debug=True,
    routes=[
        Route("/send", send_task, methods=["POST"]),
        Route("/sendSubscribe", send_task_subscribe, methods=["POST"]),
        Route("/push-receiver", push_receiver, methods=["POST"]),
        Route("/.well-known/jwks.json", auth.expose_jwks, methods=["GET"]),
    ]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", port=PORT, reload=True)
