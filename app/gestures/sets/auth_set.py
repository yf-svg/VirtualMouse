from __future__ import annotations

from app.security.auth import GestureAuthCfg

def auth_allowed_for_cfg(cfg: GestureAuthCfg) -> set[str]:
    return set(cfg.sequence) | set(cfg.reset_gestures) | set(cfg.approve_gestures) | set(cfg.back_gestures)


def auth_priority_for_cfg(cfg: GestureAuthCfg) -> list[str]:
    ordered = [
        *cfg.reset_gestures,
        *cfg.back_gestures,
        *cfg.approve_gestures,
        *reversed(cfg.sequence),
    ]
    unique: list[str] = []
    for label in ordered:
        if label not in unique:
            unique.append(label)
    return unique


_DEFAULT_AUTH_CFG = GestureAuthCfg()

# Runtime auth is intentionally strict: only gestures with explicit auth roles are
# allowed to surface during locked/auth states.
AUTH_ALLOWED = auth_allowed_for_cfg(_DEFAULT_AUTH_CFG)

# Highest priority first
AUTH_PRIORITY = auth_priority_for_cfg(_DEFAULT_AUTH_CFG)
