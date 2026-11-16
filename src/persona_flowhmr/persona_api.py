"""LAPLACE Persona Plugin API WebSocket client (`ws://127.0.0.1:25034`).

Matches the exact wire protocol in `@laplace.live/persona-sdk` (`wire/envelope.ts`, `wire/methods.ts`, `wire/types.ts`):
  - Envelope: `{"kind": "request", "id": ..., "method": ..., "params": ...}`
  - Handshake: `Authorization: Bearer <token>` + server `{"kind": "hello", "protocol": ..., "app": ...}`
  - `session.identify` (`{name, version, developer}`)
  - VMC receiver setup: `settings.get`, `tracking.addSource` / `tracking.updateSource`, `pose.setEnabled`,
    `scene.get` + `instance.setTrackingSources` (`{instanceId, poseSourceId}`)
  - Live2D parameter & input leases: `param.inject` (`{entries: [{type, id, value}]}`)
  - Stage verification: `tracking.status` & `stage.capture` (`{dataUrl, width, height}`)
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from pathlib import Path
from typing import Any

import websockets

from . import __version__


class PersonaPluginError(RuntimeError):
    """Error returned by the LAPLACE Persona Plugin API (`kind == 'error'`)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


class PersonaPluginClient:
    """Thread-safe background WebSocket client speaking Persona's exact SDK envelope."""

    def __init__(
        self,
        url: str = "ws://127.0.0.1:25034",
        token: str = "",
        *,
        app_name: str = "Persona FlowHMR",
        app_version: str = __version__,
        developer: str = "LAPLACE",
    ) -> None:
        self.url = url
        self.token = token
        self.app_name = app_name
        self.app_version = app_version
        self.developer = developer

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._req_counter = 0
        self._connected = threading.Event()
        self._stop = threading.Event()

        self.server_hello: dict[str, Any] | None = None
        self.has_live2d_model: bool = True
        self.last_error: str = ""

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self, timeout_s: float = 5.0) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return self.is_connected
        self._stop.clear()
        self._connected.clear()
        self._thread = threading.Thread(target=self._run_loop, name="persona-plugin-api", daemon=True)
        self._thread.start()
        return self._connected.wait(timeout=timeout_s)

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown_ws(), loop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._connected.clear()

    async def _shutdown_ws(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._connection_supervisor())
        finally:
            loop.close()

    async def _connection_supervisor(self) -> None:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    additional_headers=headers,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    raw_hello = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg = json.loads(raw_hello)
                    if msg.get("kind") == "hello":
                        self.server_hello = msg
                    elif msg.get("kind") == "error":
                        raise PersonaPluginError(str(msg.get("code", "AUTH")), str(msg.get("message", "")))

                    reader_task = asyncio.create_task(self._recv_loop(ws))
                    try:
                        await self._send_request_async(
                            "session.identify",
                            {
                                "name": self.app_name,
                                "version": self.app_version,
                                "developer": self.developer,
                            },
                        )
                        self.last_error = ""
                        self._connected.set()
                        await reader_task
                    finally:
                        reader_task.cancel()
                        self._connected.clear()
            except Exception as exc:
                self.last_error = str(exc)
                self._connected.clear()
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.set_exception(exc)
                self._pending.clear()
                if self._stop.is_set():
                    break
                await asyncio.sleep(2.0)

    async def _recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("kind")
            if kind == "response":
                req_id = str(msg.get("id", ""))
                fut = self._pending.pop(req_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg.get("result"))
            elif kind == "error":
                req_id = str(msg.get("id") or "")
                fut = self._pending.pop(req_id, None)
                if fut is not None and not fut.done():
                    fut.set_exception(
                        PersonaPluginError(str(msg.get("code", "UNKNOWN")), str(msg.get("message", "")))
                    )

    async def _send_request_async(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._ws is None:
            raise RuntimeError("Plugin API socket is not connected")
        self._req_counter += 1
        req_id = f"req-{self._req_counter}"
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        frame: dict[str, Any] = {"kind": "request", "id": req_id, "method": method}
        if params is not None:
            frame["params"] = params
        await self._ws.send(json.dumps(frame))
        return await asyncio.wait_for(fut, timeout=10.0)

    def call(self, method: str, params: dict[str, Any] | None = None, timeout_s: float = 10.0) -> Any:
        """Synchronously invoke a Plugin API RPC method from any thread."""
        if not self.is_connected or self._loop is None:
            raise RuntimeError("PersonaPluginClient is not connected")
        coro = self._send_request_async(method, params)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout_s)

    def ensure_vmc_receiver(
        self,
        vmc_port: int = 39539,
        *,
        bind_primary_avatar: bool = True,
        follow_movement: bool = True,
    ) -> dict[str, Any]:
        """Ensure Persona has an enabled `vmc` tracking source on `vmc_port` and bind any VRM avatar."""
        settings_res = self.call("settings.get")
        settings = settings_res.get("settings", {}) if isinstance(settings_res, dict) else {}
        sources = settings.get("tracking", {}).get("sources", [])

        vmc_src = next(
            (src for src in sources if src.get("kind") == "vmc" and int(src.get("port") or -1) == int(vmc_port)),
            None,
        )
        is_new = vmc_src is None
        if is_new:
            added = self.call(
                "tracking.addSource",
                {"kind": "vmc", "name": "FlowHMR Live Mocap", "port": int(vmc_port)},
            )
            vmc_src = added.get("source", added)
        source_id = str(vmc_src["id"])
        # A freshly added source reporting no `enabled` flag is assumed enabled; an existing one is not.
        if not vmc_src.get("enabled", is_new):
            self.call("tracking.updateSource", {"id": source_id, "enabled": True})

        # Enable network body master switch (`pose.setEnabled`).
        self.call("pose.setEnabled", {"enabled": True})

        bound_instance_id: str | None = None
        if bind_primary_avatar:
            inst_res = self.call("instance.list")
            instances = inst_res.get("instances", []) if isinstance(inst_res, dict) else []
            self.has_live2d_model = any(
                it.get("kind") == "model" and it.get("format") == "live2d" for it in instances
            )
            vrm_inst = next(
                (it for it in instances if it.get("kind") == "model" and it.get("format") == "vrm"),
                None,
            )
            if vrm_inst is not None:
                bound_instance_id = str(vrm_inst["instanceId"])
                caps = set((self.server_hello or {}).get("app", {}).get("capabilities", []))
                bind_req: dict[str, Any] = {
                    "instanceId": bound_instance_id,
                    "poseSourceId": source_id,
                }
                if "pose-movement-options" in caps:
                    bind_req["poseFollowMovement"] = bool(follow_movement)
                self.call("instance.setTrackingSources", bind_req)

        return {"sourceId": source_id, "boundInstanceId": bound_instance_id}

    def inject_params_fire_and_forget(self, entries: list[dict[str, Any]]) -> None:
        """Asynchronously inject `InjectEntry[]` (`param.inject`) without blocking the playout clock."""
        if not self.is_connected or not self.has_live2d_model or self._loop is None or not entries:
            return

        async def _send() -> None:
            try:
                await self._send_request_async("param.inject", {"entries": entries})
            except Exception as exc:
                self.last_error = str(exc)

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def capture_stage_png(self, out_path: str | Path, max_edge: int = 1280) -> Path:
        """Capture Persona's stage via `stage.capture` (`dataUrl`) and write PNG to `out_path`."""
        res = self.call("stage.capture", {"maxEdge": int(max_edge)}, timeout_s=15.0)
        data_url = str(res["dataUrl"])
        _, b64 = data_url.split(",", 1) if "," in data_url else ("", data_url)
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(b64))
        return dest
