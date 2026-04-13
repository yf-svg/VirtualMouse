# PROJECT TRACKER

## RESUME HERE
- Phase 4 next -> grouped hyperparameter search / model selection pipeline
- Then -> train/export first real runtime bundle from validated data
- Then -> Phase 5 auth on unified runtime predictions

## PROJECT SNAPSHOT
- Virtual Gesture Mouse
- Goal: ML-first gesture runtime; rules = fallback only
- Backbone: MediaPipe Hands
- State: Phase 4 scaffold DONE; no trained model yet
- Runtime: classifier seam live; actual predictions still rules via fallback
- Auth/control/presentation: stubs

## CURRENT PIPELINE
Camera -> Preprocess -> MediaPipe -> Quality gate -> Features(`phase3.v2`) -> Feature window -> Classifier(ML first) -> Rule fallback -> Temporal stable -> Overlay -> future Auth/Mode/Action

## STATUS
- camera/input: VERIFIED
- hand detection: VERIFIED
- feature extraction: VERIFIED
- rule engine: DONE
- validation engine: DONE
- tracker/readiness tools: DONE
- ML runtime seam: DONE
- trainer pipeline: DONE
- model selection pipeline: DONE
- runtime bundle export: DONE
- runtime model deployment: DONE
- runtime hysteresis/hold gate: DONE
- auth: DONE
- mode router: PARTIAL
- mouse/control: PARTIAL
- presentation mode: DONE
- fps>=25: BUG

## LOCKED RULES
- Pipeline order fixed: record -> validate -> split/CV -> train -> eval -> runtime
- ML first; rules fallback only
- `phase3.v2` feature contract must match train/runtime
- `BRAVO` = canonical thumbs-up label
- `ONE..FIVE` = auth only; never general ops
- one gesture -> one action only
- ambiguity must not trigger actions
- smoothing/jitter reduction mandatory
- future actions/auth must consume `eligible`, not raw `chosen`
- do not invent scroll/clutch/click semantics silently
- tracker must be updated after every implementation step

## GESTURE MAP
- Auth-only: `ONE TWO THREE FOUR FIVE`
- Ops runtime: `FIST CLOSED_PALM OPEN_PALM PEACE_SIGN L BRAVO THUMBS_DOWN POINT_RIGHT POINT_LEFT PINCH_INDEX PINCH_MIDDLE PINCH_RING PINCH_PINKY PINCH_IM PINCH_IMRP`
- Presentation subset: `POINT_RIGHT POINT_LEFT OPEN_PALM FIST`
- Action map: not implemented yet
- Notable constraints:
  - `BRAVO` vs `FIST` tuned
  - `THUMBS_DOWN` natural variants accepted
  - `POINT_*` uses display-space direction; palm/back is debug, not hard reject

## KNOWN ISSUES
- BUG: `app/gestures/registry.py` still short-circuits `THUMBS_DOWN` and `FIST`
- BUG: no real exported `models/gesture_svm.joblib` deployed yet; runtime still falls back to rules
- NOTE: baseline trainer writes `training_candidate`; explicit export now writes runtime-approved `runtime_model_bundle`
- NOTE: live deployment now uses explicit promotion with bundle validation + backup
- NOTE: grouped search runs on train users only; skips cleanly if train split has <2 users
- NOTE: runtime now exposes `stable` + `eligible`; `eligible` is the future action-safe signal
- NOTE: `ModeRouter` now owns auth state transitions; runtime loop delegates auth policy
- NOTE: auth now enforces retry limits with temporary lockout after repeated failures/timeouts
- NOTE: sleep->auth wake and relock semantics now live in `ModeRouter`
- NOTE: presentation mode is now app-gated through `WindowWatch` and router permission sync
- NOTE: general-mode dry run now includes a stateful `PINCH_INDEX` primary interaction path using abstract cursor-space movement; cursor pose remains decoupled/provisional
- NOTE: `FIST` clutch now owns General Mode above primary interaction, cancels lower-priority primary state, and uses the same abstract cursor-space seam
- NOTE: `PINCH_MIDDLE` right-click now runs as a sibling secondary controller on the same seam, below clutch/primary ownership and emitting only on valid release
- NOTE: `PINCH_IM` scroll mode now owns General Mode beneath clutch, uses edge-triggered toggle + axis lock on the same seam, and suppresses primary/secondary progression while active
- NOTE: cursor preview now exists as the lowest-priority General Mode owner; cursor pose policy is isolated and still provisional (`L` by default)
- NOTE: safe OS execution now uses an explicit `dry_run`/`live` profile plus per-subsystem toggles; invalid or incomplete live config fails safe to no OS action
- NOTE: runtime execution now passes through `ExecutionSafetyGate`, which suppresses unsafe live cursor/primary/secondary/scroll output on hand loss or feature instability and taints unstable primary clicks
- NOTE: presentation mode now resolves `POINT_RIGHT/POINT_LEFT/OPEN_PALM/FIST` through a dedicated policy and uses the same safety + execution seams as General Mode
- NOTE: `WindowWatch` now performs conservative foreground-app detection for PowerPoint, presentation-like browser tabs, and common PDF viewers; ambiguous context fails safe to no presentation action
- NOTE: `OPEN_PALM -> PRESENT_START` and `FIST -> PRESENT_EXIT` are now implemented as localized provisional presentation semantics because the repo had the reserved gesture subset but not the final action detail
- BUG: validation not auto-run after recording save
- BUG: FPS target not yet met
- RISK: presentation context detection is intentionally conservative; unsupported apps/titles will fail safe to General Mode routing or no action

## NEXT STEP
1. Finish remaining Phase 6 lifecycle product items: gesture/manual exit path and explicit operator override policy
2. Phase 7 focus: FPS, latency, and end-to-end runtime polish
3. Data-blocked path: train/export/promote first real runtime bundle

## AGENT RULES
- Read: `PROJECT_TRACKER.md` -> `ARCHITECTURE_DECISIONS.md` -> `INTERFACES_AND_CONTRACTS.md` -> `OPEN_ISSUES.md` -> `SESSION_LOG.md`
- Respect locked rules; mark uncertain facts as `ASSUMED`/`UNVERIFIED`
- After any code change: update tracker files immediately
