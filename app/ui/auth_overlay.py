from __future__ import annotations

import time

import cv2

from app.ui.auth_overlay_state import AuthOverlayState


class AuthOverlay:
    def __init__(self):
        self._last_sequence_len = 0
        self._last_status = "idle"
        self._anim_kind: str | None = None
        self._anim_until = 0.0
        self._frozen_sequence: tuple[str, ...] = ()

    def draw(self, frame_bgr, state: AuthOverlayState) -> None:
        now = time.perf_counter()
        target_sequence = state.current_sequence
        if state.freeze_sequence:
            if self._frozen_sequence:
                target_sequence = self._frozen_sequence
            else:
                self._frozen_sequence = state.current_sequence
        else:
            self._frozen_sequence = state.current_sequence

        self._update_animation(now=now, state=state, target_sequence=target_sequence)
        progress = self._animation_progress(now)

        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        center_y = max(120, h // 2)

        panel_h = 88
        back_w = 120
        ok_w = 120
        seq_w = 320
        gap = 18
        scale = 1.0 + (0.06 * progress if self._anim_kind == "accept" else 0.0)
        panel_h = int(round(panel_h * scale))
        seq_h = panel_h + 8

        total_w = back_w + gap + seq_w + gap + ok_w
        start_x = center_x - (total_w // 2)
        back_rect = (start_x, center_y - panel_h // 2, back_w, panel_h)
        seq_rect = (start_x + back_w + gap, center_y - seq_h // 2, seq_w, seq_h)
        ok_rect = (seq_rect[0] + seq_w + gap, center_y - panel_h // 2, ok_w, panel_h)

        self._draw_text(frame_bgr, f"Detected: {state.detected_gesture or 'NONE'}", (center_x, center_y - 78), 0.75, center=True)
        self._draw_box(frame_bgr, back_rect, fill=(26, 32, 44), border=(176, 188, 208))
        self._draw_box(frame_bgr, seq_rect, fill=(12, 18, 28), border=(220, 228, 240))
        self._draw_box(frame_bgr, ok_rect, fill=(26, 32, 44), border=(176, 188, 208))

        self._draw_text(frame_bgr, "BACK", self._rect_center(back_rect), 0.8, center=True)
        self._draw_text(frame_bgr, "OK", self._rect_center(ok_rect), 0.8, center=True)
        self._draw_sequence(frame_bgr, seq_rect, target_sequence, state.total_required, anim_kind=self._anim_kind, progress=progress)
        self._draw_text(frame_bgr, state.status_text, (center_x, center_y + 78), 0.72, center=True)

    def _update_animation(self, *, now: float, state: AuthOverlayState, target_sequence: tuple[str, ...]) -> None:
        seq_len = len(target_sequence)
        status = state.auth_status
        if status in {"reset_wrong", "reset_timeout", "reset_cancel"} and self._last_status != status:
            self._anim_kind = "reset"
            self._anim_until = now + 0.18
        elif seq_len > self._last_sequence_len:
            self._anim_kind = "accept"
            self._anim_until = now + 0.16
        elif seq_len < self._last_sequence_len and status == "step_back":
            self._anim_kind = "back"
            self._anim_until = now + 0.16
        elif now >= self._anim_until:
            self._anim_kind = None

        self._last_sequence_len = seq_len
        self._last_status = status

    def _animation_progress(self, now: float) -> float:
        if self._anim_kind is None or now >= self._anim_until:
            return 0.0
        duration = 0.18 if self._anim_kind == "reset" else 0.16
        return max(0.0, min(1.0, (self._anim_until - now) / duration))

    def _draw_sequence(self, frame_bgr, rect, sequence: tuple[str, ...], total_required: int, *, anim_kind: str | None, progress: float) -> None:
        x, y, w, h = rect
        slot_gap = 12
        slot_w = max(36, (w - 48 - ((total_required - 1) * slot_gap)) // max(1, total_required))
        slot_h = 44
        start_x = x + (w - ((slot_w * total_required) + ((total_required - 1) * slot_gap))) // 2
        slot_y = y + (h - slot_h) // 2
        digits = tuple(_to_digit(label) for label in sequence)

        for idx in range(total_required):
            filled = idx < len(digits)
            alpha = 1.0
            if anim_kind == "back" and idx == len(digits):
                alpha = 1.0 - progress
                filled = True
            elif anim_kind == "reset" and idx < len(digits):
                alpha = 1.0 - progress
            slot_x = start_x + idx * (slot_w + slot_gap)
            fill = (96, 210, 148) if filled else (44, 56, 72)
            fill = tuple(int(component * alpha) for component in fill)
            border = (226, 232, 240) if filled else (116, 130, 150)
            self._draw_box(frame_bgr, (slot_x, slot_y, slot_w, slot_h), fill=fill, border=border)
            text = digits[idx] if idx < len(digits) else "_"
            self._draw_text(frame_bgr, text, (slot_x + slot_w // 2, slot_y + slot_h // 2 + 2), 0.8, center=True)

    @staticmethod
    def _draw_box(frame_bgr, rect, *, fill, border) -> None:
        x, y, w, h = rect
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), fill, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), border, thickness=2, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_text(frame_bgr, text: str, origin, scale: float, *, center: bool = False) -> None:
        x, y = origin
        size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if center:
            x -= size[0] // 2
            y += size[1] // 2
        cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 247, 250), 2, cv2.LINE_AA)

    @staticmethod
    def _rect_center(rect) -> tuple[int, int]:
        x, y, w, h = rect
        return (x + w // 2, y + h // 2)


def _to_digit(label: str) -> str:
    return {
        "ONE": "1",
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
    }.get(label, "?")
