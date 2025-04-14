# client.py
import requests
import sseclient
import time, uuid, json, hashlib, threading
import jwt
from jwcrypto import jwk
from jwt import PyJWK
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

# ==== CONFIG ====
PORT = 8000
CLIENT_JWKS_PORT = 8002  # Where the client exposes its public key
BASE = f"http://localhost:{PORT}"

# ==== KEY GENERATION ====
key = jwk.JWK.generate(kty="RSA", size=2048, kid=str(uuid.uuid4()), use="sig")
private_key = PyJWK.from_json(key.export_private())
public_jwk_dict = key.export_public(as_dict=True)

# ==== START CLIENT JWKS SERVER ====
def expose_jwks():
    async def handle_jwks(request):
        return JSONResponse({"keys": [public_jwk_dict]})
    app = Starlette(routes=[Route("/.well-known/jwks.json", handle_jwks)])
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=CLIENT_JWKS_PORT, log_level="warning"),
        daemon=True
    ).start()

expose_jwks()
time.sleep(1)  # Give the thread a moment to start

# ==== JWT Generator ====
def generate_jwt(data):
    hashed = hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()
    return jwt.encode(
        {"iat": int(time.time()), "request_body_sha256": hashed},
        key=private_key,
        algorithm="RS256",
        headers={"kid": private_key.key_id},
    )

# ==== RPC Wrapper ====
def send_rpc(method: str, params: dict):
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": params
    }
    token = generate_jwt(payload)
    headers = {"Authorization": f"Bearer {token}"}
    return payload, headers

# ==== 1. Send ====
print("\n>>> /send")
payload, headers = send_rpc("send", {"text": "hello world"})
res = requests.post(f"{BASE}/send", json=payload, headers=headers)
print(res.json())

# # ==== 2. Stream ====
# print("\n>>> /sendSubscribe (streaming)")
# payload, headers = send_rpc("sendSubscribe", {"text": "this is streamed live"})
# res = requests.post(f"{BASE}/sendSubscribe", json=payload, headers=headers, stream=True)
# client = sseclient.SSEClient(res)
# for event in client.events():
#     print("Stream:", event.data)

# # ==== 3. Push ====
# print("\n>>> /push-receiver")
# push_payload = {"task_id": "abc123", "status": "done"}
# token = generate_jwt(push_payload)
# res = requests.post(f"{BASE}/push-receiver", json=push_payload, headers={"Authorization": f"Bearer {token}"})
# print(res.json())
