# ARCHITECTURE DECISIONS

## LOCKED
- ADR-001 | Pipeline | record -> validate -> split/CV -> train -> eval -> runtime
- ADR-002 | Perception | MediaPipe Hands is current backbone
- ADR-003 | Runtime | ML first -> rules fallback
- ADR-004 | Labels | `app/gestures/sets/labels.py` = source of truth
- ADR-005 | Auth | `ONE..FIVE` reserved for auth only
- ADR-006 | Actions | one gesture -> one action; ambiguity -> no action
- ADR-007 | Features | schema/version fixed at `phase3.v2` until explicit migration
- ADR-008 | Stability | smoothing + jitter reduction mandatory
- ADR-009 | Data ops | tracker + readiness derive from canonical labels
- ADR-011 | Runtime bundle | runtime loads only exported `runtime_model_bundle`; reject `training_candidate`
- ADR-012 | Trainer output | train -> candidate artifact in `data/datasets`; explicit export required for `models/gesture_svm.joblib`
- ADR-013 | Runtime eligibility | auth/actions must consume gated `eligible` output, not raw `chosen`

## PROVISIONAL
- ADR-010 | Presentation | app-gated via `WindowWatch`; not globally active
- ADR-014 | General dry run | explicit `ActionIntent` contract; unmapped/default no-op until control semantics are approved
- ADR-015 | Primary interaction | `PINCH_INDEX` click/drag/double-click must consume `eligible` + abstract cursor-space movement; no dependency on final cursor-pose gesture
- ADR-016 | Clutch | `FIST` clutch must sit above primary interaction on the same abstract cursor-space seam and suppress same-frame re-entry on release
- ADR-017 | Secondary interaction | `PINCH_MIDDLE` right-click must use a sibling release-only controller on the same seam; clutch/primary ownership suppress it
- ADR-018 | Scroll mode | scroll must use a dedicated non-pinch toggle gesture (current default: `SHAKA`) as an edge-triggered mode controller below clutch and above primary/secondary, with deterministic axis lock/reset on the same seam
- ADR-019 | Cursor policy | cursor pose choice remains provisional and must stay isolated in a dedicated policy layer; movement preview/output lives in a separate controller beneath higher-priority owners
- ADR-020 | OS execution | OS side effects must live in a separate execution adapter behind the dry-run contract; live cursor reuses cursor preview output and remains globally disableable
- ADR-021 | General live execution | primary/secondary/scroll live actions must be translated only from dry-run subsystem outputs; drag cursor motion reuses the primary subsystem cursor-space seam rather than bypassing controllers
- ADR-022 | Execution safety | base execution config remains `dry_run`, but the repo-default runtime may resolve through a centralized `fallback_live` operator override; live enablement must stay explicit, auditable, and subject to runtime safety suppression before actions reach the executor
- ADR-023 | Presentation mode | presentation actions must use a dedicated mode-aware resolver plus conservative foreground-app context detection; unresolved `OPEN_PALM`/`PEACE_SIGN` semantics stay localized and provisional rather than leaking into General Mode
- ADR-024 | Operator lifecycle | runtime exit must flow through an explicit lifecycle seam that neutralizes held OS/controller state before entering `EXITING`; manual exit remains primary and gesture exit stays localized/provisional
- ADR-025 | Override policy | execution/routing overrides must be centralized, fail-safe, and layered into the existing router + executor seams; overrides may tighten behavior but must not bypass dry-run ownership or execution safety gating

## REJECTED
- R-001 | pure rule runtime as final architecture
- R-002 | training/runtime from raw recordings without validation
- R-003 | number gestures in general ops
- R-004 | silent invention of scroll/clutch/cursor semantics

## DO NOT CHANGE WITHOUT APPROVAL
- pipeline order
- `BRAVO` naming
- auth-only numbers
- `phase3.v2` contract
- ML-first + rule-fallback structure
- runtime use of non-exported candidate artifacts
- using raw `chosen` as action/auth trigger
- one-gesture-one-action
- ambiguity -> no action
- smoothing mandatory
