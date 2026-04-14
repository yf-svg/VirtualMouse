from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2

from app.config import CONFIG
from app.constants import AppState
from app.control.actions import format_action_intent
from app.control.clutch import ClutchController
from app.control.execution import ExecutionBatchReport, OSActionExecutor
from app.control.execution_safety import ExecutionSafetyGate
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import cursor_point_from_landmarks
from app.control.primary_interaction import PrimaryInteractionController
from app.control.secondary_interaction import SecondaryInteractionController
from app.control.scroll_mode import ScrollModeController
from app.control.window_watch import PresentationContext, WindowWatch
from app.gestures.sets.auth_set import auth_allowed_for_cfg, auth_priority_for_cfg
from app.gestures.suite import GestureSuite
from app.lifecycle.operator_lifecycle import OperatorLifecycleController, neutralize_runtime_ownership
from app.lifecycle.operator_policy import ResolvedOperatorOverridePolicy, resolve_operator_override_policy
from app.lifecycle.runtime_status import _format_execution_policy_status, _format_presentation_context
from app.modes.general import resolve_general_action
from app.modes.router import ModeRouter
from app.modes.presentation import PresentationModeOut, resolve_presentation_action
from app.perception.camera import Camera
from app.perception.hand_tracker import HandTracker
from app.perception.landmark_smoothing import SelectiveLandmarkSmoother
from app.perception.preprocessing import Preprocessor
from app.perception.threaded_camera import ThreadedCamera
from app.security.auth_runtime import AuthGestureInterpreter
from app.ui.auth_overlay_state import AuthOverlayStateStore
from app.ui.overlay import Overlay
from app.utils.fps import FPSCounter


@dataclass(slots=True)
class RuntimeContext:
    cam: Camera
    cam_thread: ThreadedCamera
    pre: Preprocessor
    tracker: HandTracker
    smoother: SelectiveLandmarkSmoother
    auth_suite: GestureSuite
    auth_interpreter: AuthGestureInterpreter
    ops_suite: GestureSuite
    router: ModeRouter
    clutch: ClutchController
    scroll_mode: ScrollModeController
    primary_interaction: PrimaryInteractionController
    secondary_interaction: SecondaryInteractionController
    cursor_preview: CursorPreviewController
    operator_lifecycle: OperatorLifecycleController
    override_policy: ResolvedOperatorOverridePolicy
    executor: OSActionExecutor
    execution_safety: ExecutionSafetyGate
    window_watch: WindowWatch
    auth_overlay_store: AuthOverlayStateStore
    overlay: Overlay
    fps_counter: FPSCounter
    window_name: str
    detect_scale: float
    stride: int


@dataclass(slots=True)
class RuntimeState:
    frame_i: int = 0
    last_hand: Any | None = None
    last_seq_processed: int = -1
    last_frame_disp: Any | None = None
    last_extra: str = "Starting..."
    cam_frames: int = 0
    cam_fps: float = 0.0
    cam_fps_t0: float = 0.0
    auth_status: str = "idle"
    auth_progress_text: str = "Auth starting"
    lifecycle_status_text: str = "LIFE:ready"


def _mirror_landmarks(landmarks):
    lm2 = type(landmarks)()
    lm2.CopyFrom(landmarks)
    for p in lm2.landmark:
        p.x = 1.0 - p.x
    return lm2


def _build_runtime_context(initial_state: AppState) -> RuntimeContext:
    router = ModeRouter(initial_state=initial_state)
    auth_allowed = auth_allowed_for_cfg(router.auth_cfg)
    auth_priority = auth_priority_for_cfg(router.auth_cfg)
    cam = Camera(
        device_index=CONFIG.camera_index,
        width=CONFIG.cam_width,
        height=CONFIG.cam_height,
    )
    pre = Preprocessor(enable=CONFIG.enable_preprocessing)
    tracker = HandTracker(max_num_hands=1)
    smoother = SelectiveLandmarkSmoother(
        strong_min_cutoff=1.6,
        strong_beta=0.18,
        tip_min_cutoff=3.5,
        tip_beta=0.45,
        d_cutoff=1.0,
    )
    auth_suite = GestureSuite(allowed=auth_allowed, priority=auth_priority)
    auth_interpreter = AuthGestureInterpreter(auth_cfg=router.auth_cfg)
    ops_suite = GestureSuite()
    clutch = ClutchController()
    scroll_mode = ScrollModeController()
    primary_interaction = PrimaryInteractionController()
    secondary_interaction = SecondaryInteractionController()
    cursor_preview = CursorPreviewController()
    override_policy = resolve_operator_override_policy()
    operator_lifecycle = OperatorLifecycleController()
    executor = OSActionExecutor(cfg=override_policy.effective_execution)
    execution_safety = ExecutionSafetyGate()
    window_watch = WindowWatch()
    auth_overlay_store = AuthOverlayStateStore()
    overlay = Overlay()
    fps_counter = FPSCounter(avg_window=30)

    window_name = f"{CONFIG.app_name} - Gesture Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, CONFIG.cam_width, CONFIG.cam_height)
    except Exception:
        pass

    cam.open()
    cam_thread = ThreadedCamera(cam.cap)
    cam_thread.start()

    return RuntimeContext(
        cam=cam,
        cam_thread=cam_thread,
        pre=pre,
        tracker=tracker,
        smoother=smoother,
        auth_suite=auth_suite,
        auth_interpreter=auth_interpreter,
        ops_suite=ops_suite,
        router=router,
        clutch=clutch,
        scroll_mode=scroll_mode,
        primary_interaction=primary_interaction,
        secondary_interaction=secondary_interaction,
        cursor_preview=cursor_preview,
        operator_lifecycle=operator_lifecycle,
        override_policy=override_policy,
        executor=executor,
        execution_safety=execution_safety,
        window_watch=window_watch,
        auth_overlay_store=auth_overlay_store,
        overlay=overlay,
        fps_counter=fps_counter,
        window_name=window_name,
        detect_scale=CONFIG.detect_scale,
        stride=max(1, CONFIG.inference_stride),
    )


def _initial_runtime_state() -> RuntimeState:
    return RuntimeState(cam_fps_t0=time.perf_counter())


def _update_camera_fps(seq: int, state: RuntimeState) -> None:
    if seq != state.last_seq_processed:
        state.cam_frames += 1

    now = time.perf_counter()
    if now - state.cam_fps_t0 >= 1.0:
        state.cam_fps = state.cam_frames / (now - state.cam_fps_t0)
        state.cam_frames = 0
        state.cam_fps_t0 = now


def _format_hand_overlay(handed: str | None, out) -> str:
    source = out.source.upper()
    confidence = f"{out.confidence:.2f}" if out.confidence is not None else "-"
    return (
        f"Hand:{handed or 'Unknown'} | "
        f"SRC:{source}({confidence}) | "
        f"CHOSEN:{out.chosen or 'NONE'} | "
        f"STABLE:{out.stable or 'NONE'} | "
        f"READY:{out.eligible or 'NONE'} | "
        f"CAND:{','.join(sorted(out.candidates)) if out.candidates else 'NONE'} | "
        f"HOLD:{out.hold_frames} | "
        f"EDGE:+{out.down or '-'} -{out.up or '-'} | "
        f"WHY:{out.reason} | GATE:{out.gate_reason}"
    )


def _format_locked_overlay(handed: str | None, out, runtime_state: RuntimeState) -> str:
    return f"{_format_hand_overlay(handed, out)} | AUTH:{runtime_state.auth_status} | {runtime_state.auth_progress_text}"


def _format_execution_report(report: ExecutionBatchReport) -> str:
    parts: list[str] = []
    for name, item in (
        ("CUR", report.cursor),
        ("PRI", report.primary),
        ("SEC", report.secondary),
        ("SCR", report.scroll),
    ):
        if item.performed:
            label = item.reason
            if item.target is not None:
                label = f"{label}@{item.target.x},{item.target.y}"
            parts.append(f"{name}:{label}")
    if not parts:
        for name, item in (
            ("CUR", report.cursor),
            ("PRI", report.primary),
            ("SEC", report.secondary),
            ("SCR", report.scroll),
        ):
            if item.reason not in {"execution_disabled", "primary_no_action", "secondary_no_action", "scroll_no_action"}:
                parts.append(f"{name}:{item.reason}")
                break
    if not parts:
        parts.append("idle")
    return f"EXEC:{'|'.join(parts)}"


def _format_single_execution_report(report) -> str:
    if report.performed:
        return f"EXEC:{report.reason}"
    return f"EXEC:{report.reason}"


def _request_exit_from_key(ctx: RuntimeContext, key: int):
    return ctx.operator_lifecycle.request_from_key(key)


def _perform_exit(
    ctx: RuntimeContext,
    state: RuntimeState,
    request,
    *,
    frame_disp=None,
    base_extra: str = "",
) -> None:
    neutralization = neutralize_runtime_ownership(ctx, reason=request.reason)
    route = ctx.router.request_exit(source=request.source, reason=request.trigger)
    state.auth_status = route.auth_status
    state.auth_progress_text = route.auth_progress_text
    state.lifecycle_status_text = ctx.operator_lifecycle.status_text(
        request=request,
        neutralization=neutralization,
    )
    extra = state.lifecycle_status_text if not base_extra else f"{base_extra} | {state.lifecycle_status_text}"
    state.last_extra = extra

    if frame_disp is not None:
        frame_out = frame_disp.copy()
        fps = ctx.fps_counter.tick()
        ctx.overlay.draw(
            frame_out,
            state_text=ctx.router.state.value,
            fps=fps,
            extra=f"{extra} | CAM_FPS:{state.cam_fps:.1f}",
        )
        state.last_frame_disp = frame_out
        cv2.imshow(ctx.window_name, frame_out)
        cv2.waitKey(1)

def run_loop(initial_state: AppState) -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    ctx = _build_runtime_context(initial_state)
    state = _initial_runtime_state()

    try:
        while True:
            frame_raw, seq, _ts, _ok = ctx.cam_thread.read_latest()

            if frame_raw is None:
                key = cv2.waitKey(1) & 0xFF
                request = _request_exit_from_key(ctx, key)
                if request is not None:
                    _perform_exit(ctx, state, request)
                    break
                continue

            _update_camera_fps(seq, state)

            if seq == state.last_seq_processed and state.last_frame_disp is not None:
                disp = state.last_frame_disp.copy()
                fps = ctx.fps_counter.tick()
                ctx.overlay.draw(
                    disp,
                    state_text=ctx.router.state.value,
                    fps=fps,
                    extra=f"{state.last_extra} | CAM_FPS:{state.cam_fps:.1f}",
                )
                cv2.imshow(ctx.window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                request = _request_exit_from_key(ctx, key)
                if request is not None:
                    _perform_exit(ctx, state, request, frame_disp=disp, base_extra=state.last_extra)
                    break
                continue

            state.last_seq_processed = seq
            frame_now = time.monotonic()

            frame_disp = cv2.flip(frame_raw, 1) if CONFIG.mirror_view else frame_raw

            small = cv2.resize(
                frame_raw,
                (0, 0),
                fx=ctx.detect_scale,
                fy=ctx.detect_scale,
                interpolation=cv2.INTER_AREA,
            )
            if CONFIG.enable_preprocessing:
                small = ctx.pre.apply(small)

            if (state.frame_i % ctx.stride) == 0:
                state.last_hand = ctx.tracker.detect(small)
            state.frame_i += 1

            presentation_context = ctx.window_watch.presentation_context()
            route_decision = ctx.override_policy.route_presentation(presentation_context)
            ctx.router.sync_presentation_permission(route_decision.presentation_allowed)

            if state.last_hand:
                lm_raw = state.last_hand.landmarks
                handed = state.last_hand.handedness
                lm_smooth = ctx.smoother.apply(lm_raw)
                auth_overlay_state = None

                if ctx.router.state in {AppState.IDLE_LOCKED, AppState.AUTHENTICATING}:
                    out = ctx.auth_suite.detect(state.last_hand)
                    auth_runtime = ctx.auth_interpreter.update(
                        suite_out=out,
                        expected_next=ctx.router.current_auth_expected_next,
                        hand_present=True,
                    )
                    route = ctx.router.route_auth_edge(auth_runtime.event_label, now=frame_now)
                    state.auth_status = route.auth_status
                    state.auth_progress_text = route.auth_progress_text
                    auth_overlay_state = ctx.auth_overlay_store.build_state(
                        cfg=ctx.router.auth_cfg,
                        auth_out=route.auth_out,
                        auth_status=route.auth_status,
                        detected_gesture=auth_runtime.detected_gesture,
                    )
                    extra = _format_locked_overlay(handed, out, state)
                    if route.suite_key == "ops":
                        ctx.auth_suite.reset()
                        ctx.auth_interpreter.reset()
                        ctx.ops_suite.reset()
                    ctx.clutch.reset()
                    ctx.scroll_mode.reset()
                    ctx.primary_interaction.reset()
                    ctx.secondary_interaction.reset()
                    ctx.cursor_preview.reset()
                    ctx.execution_safety.reset()
                else:
                    ctx.auth_interpreter.reset()
                    ctx.auth_overlay_store.reset()
                    out = ctx.ops_suite.detect(state.last_hand)
                    exit_request = ctx.operator_lifecycle.request_from_suite_out(
                        suite_out=out,
                        router_state=ctx.router.state,
                    )
                    if exit_request is not None:
                        _perform_exit(
                            ctx,
                            state,
                            exit_request,
                            frame_disp=frame_disp,
                            base_extra=(
                                f"{_format_hand_overlay(handed, out)} | "
                                f"{_format_presentation_context(presentation_context)} | "
                                f"{_format_execution_policy_status(ctx, 'exit_pending', route_summary=route_decision.summary())}"
                            ),
                        )
                        break
                    cursor_point = cursor_point_from_landmarks(lm_smooth)
                    if ctx.router.state == AppState.ACTIVE_PRESENTATION:
                        ctx.clutch.reset()
                        ctx.scroll_mode.reset()
                        ctx.primary_interaction.reset()
                        ctx.secondary_interaction.reset()
                        ctx.cursor_preview.reset()
                        ctx.execution_safety.reset()
                        presentation_out = resolve_presentation_action(
                            gesture_label=out.eligible,
                            context=presentation_context,
                        )
                        safety = ctx.execution_safety.evaluate_presentation(
                            suite_out=out,
                            presentation_out=presentation_out,
                            hand_present=True,
                        )
                        action_intent = presentation_out.intent
                        exec_report = ctx.executor.apply_presentation_mode(
                            presentation_out,
                            allow=safety.allow,
                            suppress_reason=safety.reason,
                        )
                        extra = (
                            f"{_format_hand_overlay(handed, out)} | "
                            f"{_format_presentation_context(presentation_context)} | "
                            f"CLT:{ctx.clutch.state.value} | "
                            f"SCR:{ctx.scroll_mode.state.value} | "
                            f"PRI:{ctx.primary_interaction.state.value} | "
                            f"SEC:{ctx.secondary_interaction.state.value} | "
                            f"CUR:{ctx.cursor_preview.state.value} | "
                            f"{format_action_intent(action_intent)} | "
                            f"{_format_execution_policy_status(ctx, safety.reason, route_summary=route_decision.summary())} | "
                            f"{state.lifecycle_status_text} | "
                            f"{_format_single_execution_report(exec_report)}"
                        )
                    else:
                        general_out = resolve_general_action(
                            gesture_label=out.eligible,
                            cursor_point=cursor_point,
                            clutch_controller=ctx.clutch,
                            scroll_controller=ctx.scroll_mode,
                            primary_controller=ctx.primary_interaction,
                            secondary_controller=ctx.secondary_interaction,
                            cursor_controller=ctx.cursor_preview,
                            now=frame_now,
                        )
                        safety = ctx.execution_safety.evaluate(
                            suite_out=out,
                            general_out=general_out,
                            hand_present=True,
                        )
                        action_intent = general_out.intent
                        exec_report = ctx.executor.apply_general_mode(general_out, safety=safety)
                        cursor_preview = general_out.cursor.preview_point
                        cursor_preview_text = "CUR:None"
                        if cursor_preview is not None:
                            cursor_preview_text = f"CURPREV:{cursor_preview.x:.2f},{cursor_preview.y:.2f}"
                        extra = (
                            f"{_format_hand_overlay(handed, out)} | "
                            f"CLT:{general_out.clutch.state.value} | "
                            f"SCR:{general_out.scroll.state.value}/{general_out.scroll.axis.value} | "
                            f"PRI:{general_out.primary.state.value} | "
                            f"SEC:{general_out.secondary.state.value} | "
                            f"CUR:{general_out.cursor.state.value}({general_out.cursor.policy.gesture_label or '-'}) | "
                            f"{cursor_preview_text} | "
                            f"{format_action_intent(action_intent)} | "
                            f"{_format_execution_policy_status(ctx, safety.summary(), route_summary=route_decision.summary())} | "
                            f"{state.lifecycle_status_text} | "
                            f"{_format_execution_report(exec_report)}"
                        )
                lm_draw = _mirror_landmarks(lm_smooth) if CONFIG.mirror_view else lm_smooth
                ctx.tracker.draw(frame_disp, lm_draw)
            else:
                ctx.smoother.reset()
                auth_overlay_state = None
                if ctx.router.state in {AppState.IDLE_LOCKED, AppState.AUTHENTICATING}:
                    ctx.auth_suite.reset()
                    ctx.auth_interpreter.update(
                        suite_out=None,
                        expected_next=ctx.router.current_auth_expected_next,
                        hand_present=False,
                    )
                    route = ctx.router.route_auth_edge(None, now=frame_now)
                    state.auth_status = route.auth_status
                    state.auth_progress_text = route.auth_progress_text
                    auth_overlay_state = ctx.auth_overlay_store.build_state(
                        cfg=ctx.router.auth_cfg,
                        auth_out=route.auth_out,
                        auth_status=route.auth_status,
                        detected_gesture=None,
                    )
                    extra = f"No hand detected | AUTH:{state.auth_status} | {state.auth_progress_text}"
                    ctx.clutch.reset()
                    ctx.scroll_mode.reset()
                    ctx.primary_interaction.reset()
                    ctx.secondary_interaction.reset()
                    ctx.cursor_preview.reset()
                    ctx.execution_safety.reset()
                elif ctx.router.state == AppState.ACTIVE_GENERAL:
                    ctx.auth_interpreter.reset()
                    ctx.auth_overlay_store.reset()
                    ctx.ops_suite.reset()
                    general_out = resolve_general_action(
                        gesture_label=None,
                        cursor_point=None,
                        clutch_controller=ctx.clutch,
                        scroll_controller=ctx.scroll_mode,
                        primary_controller=ctx.primary_interaction,
                        secondary_controller=ctx.secondary_interaction,
                        cursor_controller=ctx.cursor_preview,
                        now=frame_now,
                    )
                    safety = ctx.execution_safety.evaluate(
                        suite_out=None,
                        general_out=general_out,
                        hand_present=False,
                    )
                    exec_report = ctx.executor.apply_general_mode(general_out, safety=safety)
                    cursor_preview = general_out.cursor.preview_point
                    cursor_preview_text = "CUR:None"
                    if cursor_preview is not None:
                        cursor_preview_text = f"CURPREV:{cursor_preview.x:.2f},{cursor_preview.y:.2f}"
                    extra = (
                        f"No hand detected | CLT:{general_out.clutch.state.value} | "
                        f"SCR:{general_out.scroll.state.value}/{general_out.scroll.axis.value} | "
                        f"PRI:{general_out.primary.state.value} | "
                        f"SEC:{general_out.secondary.state.value} | "
                        f"CUR:{general_out.cursor.state.value} | "
                        f"{cursor_preview_text} | "
                        f"{format_action_intent(general_out.intent)} | "
                        f"{_format_execution_policy_status(ctx, safety.summary(), route_summary=route_decision.summary())} | "
                        f"{state.lifecycle_status_text} | "
                        f"{_format_execution_report(exec_report)}"
                    )
                elif ctx.router.state == AppState.ACTIVE_PRESENTATION:
                    ctx.auth_interpreter.reset()
                    ctx.auth_overlay_store.reset()
                    ctx.ops_suite.reset()
                    ctx.clutch.reset()
                    ctx.scroll_mode.reset()
                    ctx.primary_interaction.reset()
                    ctx.secondary_interaction.reset()
                    ctx.cursor_preview.reset()
                    presentation_out = resolve_presentation_action(
                        gesture_label=None,
                        context=presentation_context,
                    )
                    safety = ctx.execution_safety.evaluate_presentation(
                        suite_out=None,
                        presentation_out=presentation_out,
                        hand_present=False,
                    )
                    exec_report = ctx.executor.apply_presentation_mode(
                        presentation_out,
                        allow=safety.allow,
                        suppress_reason=safety.reason,
                    )
                    extra = (
                        f"No hand detected | "
                        f"{_format_presentation_context(presentation_context)} | "
                        f"{format_action_intent(presentation_out.intent)} | "
                        f"{_format_execution_policy_status(ctx, safety.reason, route_summary=route_decision.summary())} | "
                        f"{state.lifecycle_status_text} | "
                        f"{_format_single_execution_report(exec_report)}"
                    )
                else:
                    ctx.auth_interpreter.reset()
                    ctx.auth_overlay_store.reset()
                    ctx.ops_suite.reset()
                    ctx.clutch.reset()
                    ctx.scroll_mode.reset()
                    ctx.primary_interaction.reset()
                    ctx.secondary_interaction.reset()
                    ctx.cursor_preview.reset()
                    ctx.execution_safety.reset()
                    extra = "No hand detected"

            fps = ctx.fps_counter.tick()

            frame_out = frame_disp.copy()
            ctx.overlay.draw(
                frame_out,
                state_text=ctx.router.state.value,
                fps=fps,
                extra=f"{extra} | CAM_FPS:{state.cam_fps:.1f}",
                auth_overlay_state=auth_overlay_state,
            )

            state.last_frame_disp = frame_out
            state.last_extra = extra

            cv2.imshow(ctx.window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            request = _request_exit_from_key(ctx, key)
            if request is not None:
                _perform_exit(ctx, state, request, frame_disp=frame_out, base_extra=extra)
                break

    finally:
        try:
            neutralize_runtime_ownership(ctx, reason="runtime_shutdown")
        except Exception:
            pass
        ctx.cam_thread.stop()
        ctx.tracker.close()
        ctx.cam.release()
        cv2.destroyAllWindows()
