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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets

from .camera_tracker import camera_to_persona_vrm_camera

if TYPE_CHECKING:
    from .camera_tracker import CameraPoseFrame

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
        self.has_vrm_model: bool = True
        self.bound_vrm_instance_id: str | None = None
        self.bound_vrm_placement: dict[str, Any] | None = None
        self.vrm_placement_valid: bool = False
        self._placement_gen: int = 0
        self._last_placement_refresh_mono: float = 0.0
        self.camera_patches_sent: int = 0
        self._camera_inflight: bool = False
        self._pending_vrm_camera: dict[str, Any] | None = None
        self._pending_camera_pose: tuple[CameraPoseFrame, bool] | None = None
        self._last_sent_vrm_camera: dict[str, Any] | None = None
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
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
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
                    self._last_sent_vrm_camera = None
                    self._placement_gen += 1
                    self.bound_vrm_placement = None
                    self.vrm_placement_valid = False
                    self._last_placement_refresh_mono = 0.0
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
                        if self._pending_camera_pose is not None or self._pending_vrm_camera is not None:
                            asyncio.create_task(self._drain_camera_queue())
                        await reader_task
                    finally:
                        reader_task.cancel()
                        self._connected.clear()
                        self._ws = None
                        disc_err = ConnectionError("Plugin API WebSocket disconnected")
                        for fut in list(self._pending.values()):
                            if not fut.done():
                                fut.set_exception(disc_err)
                        self._pending.clear()
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
            elif kind == "event":
                # Any server-pushed scene/instance event invalidates cached and in-flight VRM placement immediately.
                self._placement_gen += 1
                self.bound_vrm_placement = None
                self.vrm_placement_valid = False
                self._last_placement_refresh_mono = 0.0

    async def _send_request_async(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 10.0,
    ) -> Any:
        if self._ws is None:
            raise RuntimeError("Plugin API socket is not connected")
        self._req_counter += 1
        req_id = f"req-{self._req_counter}"
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        frame: dict[str, Any] = {"kind": "request", "id": req_id, "method": method}
        if params is not None:
            frame["params"] = params
        try:
            await self._ws.send(json.dumps(frame))
            return await asyncio.wait_for(fut, timeout=max(0.05, float(timeout_s)))
        finally:
            self._pending.pop(req_id, None)

    def call(self, method: str, params: dict[str, Any] | None = None, timeout_s: float = 10.0) -> Any:
        """Synchronously invoke a Plugin API RPC method from any thread."""
        if not self.is_connected or self._loop is None:
            raise RuntimeError("PersonaPluginClient is not connected")
        coro = self._send_request_async(method, params, timeout_s=timeout_s)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=max(0.1, float(timeout_s) + 0.5))
        except BaseException:
            fut.cancel()
            raise

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
            start_gen = self._placement_gen
            inst_res = self.call("instance.list")
            instances = inst_res.get("instances", []) if isinstance(inst_res, dict) else []
            self.has_live2d_model = any(
                it.get("kind") == "model" and it.get("format") == "live2d" for it in instances
            )
            vrm_inst = next(
                (it for it in instances if it.get("kind") == "model" and it.get("format") == "vrm"),
                None,
            )
            self.has_vrm_model = vrm_inst is not None
            self.bound_vrm_placement = None
            self.vrm_placement_valid = False
            if vrm_inst is not None:
                bound_instance_id = str(vrm_inst["instanceId"])
                self.bound_vrm_instance_id = bound_instance_id
                caps = set((self.server_hello or {}).get("app", {}).get("capabilities", []))
                bind_req: dict[str, Any] = {
                    "instanceId": bound_instance_id,
                    "poseSourceId": source_id,
                }
                if "pose-movement-options" in caps:
                    bind_req["poseFollowMovement"] = bool(follow_movement)
                self.call("instance.setTrackingSources", bind_req)
                try:
                    inst_detail = self.call("instance.get", {"instanceId": bound_instance_id})
                    vrm_place = (inst_detail or {}).get("item", {}).get("vrm") if isinstance(inst_detail, dict) else None
                    if self._placement_gen == start_gen:
                        self.bound_vrm_placement = vrm_place if isinstance(vrm_place, dict) else {}
                        self.vrm_placement_valid = True
                        self._last_placement_refresh_mono = time.monotonic()
                except Exception as exc:
                    self.last_error = str(exc)
            else:
                self.bound_vrm_instance_id = None
                if self._placement_gen == start_gen:
                    self.vrm_placement_valid = True
                    self._last_placement_refresh_mono = time.monotonic()

        return {"sourceId": source_id, "boundInstanceId": bound_instance_id}

    async def _refresh_vrm_placement_async(self) -> bool:
        """Query primary VRM instance placement asynchronously; discards stale results invalidated by mid-RPC events."""
        for _attempt in range(2):
            start_gen = self._placement_gen
            self.bound_vrm_placement = None
            self.vrm_placement_valid = False
            try:
                inst_res = await self._send_request_async("instance.list")
                if self._placement_gen != start_gen:
                    continue
                instances = inst_res.get("instances", []) if isinstance(inst_res, dict) else []
                self.has_live2d_model = any(
                    it.get("kind") == "model" and it.get("format") == "live2d" for it in instances
                )
                vrm_inst = next(
                    (it for it in instances if it.get("kind") == "model" and it.get("format") == "vrm"),
                    None,
                )
                self.has_vrm_model = vrm_inst is not None
                if vrm_inst is None:
                    if self._placement_gen != start_gen:
                        continue
                    self.bound_vrm_instance_id = None
                    self.vrm_placement_valid = True
                    self._last_placement_refresh_mono = time.monotonic()
                    return True
                inst_id = str(vrm_inst["instanceId"])
                self.bound_vrm_instance_id = inst_id
                inst_detail = await self._send_request_async("instance.get", {"instanceId": inst_id})
                if self._placement_gen != start_gen:
                    continue
                vrm_place = (inst_detail or {}).get("item", {}).get("vrm") if isinstance(inst_detail, dict) else None
                self.bound_vrm_placement = vrm_place if isinstance(vrm_place, dict) else {}
                self.vrm_placement_valid = True
                self._last_placement_refresh_mono = time.monotonic()
                return True
            except Exception as exc:
                self.last_error = str(exc)
                return False
        return False

    def refresh_vrm_placement(self, timeout_s: float = 5.0) -> bool:
        """Synchronously refresh primary VRM stage placement (used even when VMC auto-configure is disabled)."""
        if not self.is_connected or self._loop is None:
            return False
        fut = asyncio.run_coroutine_threadsafe(self._refresh_vrm_placement_async(), self._loop)
        try:
            return bool(fut.result(timeout=timeout_s))
        except Exception as exc:
            self.last_error = str(exc)
            return False

    async def _drain_camera_queue(self) -> None:
        """Single coalesced worker draining both `sync_camera_pose_fire_and_forget` and `send_vrm_camera_fire_and_forget`."""
        if self._camera_inflight:
            return
        self._camera_inflight = True
        try:
            while self.is_connected and not self._stop.is_set():
                if self._pending_camera_pose is not None:
                    nxt_pose, nxt_mirror = self._pending_camera_pose
                    self._pending_camera_pose = None
                    placement_snapshot: dict[str, Any] | None = None
                    placement_ready = False
                    for _attempt in range(2):
                        snap_gen = self._placement_gen
                        now_m = time.monotonic()
                        if not self.vrm_placement_valid or (now_m - self._last_placement_refresh_mono) >= 1.0:
                            if not await self._refresh_vrm_placement_async():
                                break
                            snap_gen = self._placement_gen
                        if not self.has_vrm_model or not self.vrm_placement_valid:
                            break
                        if self._placement_gen != snap_gen:
                            continue
                        placement_snapshot = (
                            dict(self.bound_vrm_placement) if isinstance(self.bound_vrm_placement, dict) else None
                        )
                        if self._placement_gen == snap_gen and self.vrm_placement_valid:
                            placement_ready = True
                            break
                    if not placement_ready:
                        break
                    nxt_cam: dict[str, Any] | None = camera_to_persona_vrm_camera(
                        nxt_pose,
                        mirror=nxt_mirror,
                        vrm_placement=placement_snapshot,
                    )
                    used_placement_gen: int | None = snap_gen
                elif self._pending_vrm_camera is not None:
                    nxt_cam = self._pending_vrm_camera
                    self._pending_vrm_camera = None
                    used_placement_gen = None
                else:
                    break

                if used_placement_gen is not None and self._placement_gen != used_placement_gen:
                    # Placement was invalidated before sending; re-queue the latest pose for fresh conversion.
                    if self._pending_camera_pose is None and self._pending_vrm_camera is None:
                        self._pending_camera_pose = (nxt_pose, nxt_mirror)
                    continue
                if not nxt_cam or nxt_cam == self._last_sent_vrm_camera:
                    continue
                try:
                    await self._send_request_async("scene.patch", {"vrmCamera": nxt_cam})
                    self._last_sent_vrm_camera = nxt_cam
                    self.camera_patches_sent += 1
                except Exception as exc:
                    self.last_error = str(exc)
                    break
        finally:
            self._camera_inflight = False

    def sync_camera_pose_fire_and_forget(self, cam_pose: CameraPoseFrame, *, mirror: bool = False) -> None:
        """Refresh VRM stage placement when needed, transform `cam_pose`, and coalesce `scene.patch -> vrmCamera`."""
        if not self.is_connected or self._loop is None:
            return

        async def _enqueue_pose() -> None:
            self._pending_camera_pose = (cam_pose, bool(mirror))
            self._pending_vrm_camera = None
            await self._drain_camera_queue()

        asyncio.run_coroutine_threadsafe(_enqueue_pose(), self._loop)

    def send_vrm_camera_fire_and_forget(self, vrm_camera: dict[str, Any]) -> None:
        """Stream `scene.patch -> vrmCamera` with single-slot in-flight coalescing (zero WS backlog)."""
        if not self.is_connected or not self.has_vrm_model or self._loop is None or not vrm_camera:
            return

        async def _enqueue() -> None:
            if not self._camera_inflight and vrm_camera == self._last_sent_vrm_camera:
                return
            self._pending_vrm_camera = vrm_camera
            self._pending_camera_pose = None
            await self._drain_camera_queue()

        asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)

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
