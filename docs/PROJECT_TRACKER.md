# PROJECT TRACKER

## RESUME HERE
- Phase 4 next -> train/export first real runtime bundle from validated data
- Then -> Phase 7 runtime polish: FPS, latency, and live-feel tuning
- Then -> optional operator tooling on top of the existing override/lifecycle seams

## PROJECT SNAPSHOT
- Virtual Gesture Mouse
- Goal: ML-first gesture runtime; rules = fallback only
- Backbone: MediaPipe Hands
- State: Phase 6 runtime semantics DONE; Phase 7 polish active; no trained runtime bundle promoted yet
- Runtime: classifier seam live; actual predictions still rules via fallback until a real bundle is trained/exported/promoted
- Auth/control/presentation: implemented on the shared router/safety/execution seams

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
- Auth-only default: `ONE TWO THREE`
- Ops runtime: `FIST CLOSED_PALM OPEN_PALM SHAKA PEACE_SIGN L BRAVO THUMBS_DOWN POINT_RIGHT POINT_LEFT PINCH_INDEX PINCH_MIDDLE PINCH_RING PINCH_PINKY PINCH_IM PINCH_IMRP`
- Presentation subset: `POINT_RIGHT POINT_LEFT OPEN_PALM PEACE_SIGN`
- Action map: not implemented yet
- Notable constraints:
  - `BRAVO` vs `FIST` tuned
  - `THUMBS_DOWN` natural variants accepted
  - `POINT_*` uses display-space direction; palm/back is debug, not hard reject

## KNOWN ISSUES
- BUG: `app/gestures/registry.py` still short-circuits `THUMBS_DOWN` and `FIST`
- BUG: no real exported `models/gesture_svm.joblib` deployed yet; runtime still falls back to rules
- NOTE: baseline trainer writes `training_candidate`; explicit export now writes runtime-approved `runtime_model_bundle`
- NOTE: validation split planning now fails safe unless every dataset label can remain represented in `train`; `split_status == ok` is no longer a best-effort label mix
- NOTE: validation split planning now uses exhaustive search for small user counts and deterministic beam search for larger user counts; planner choice + assignment score are recorded in `split_plan`
- NOTE: trainer model-selection now records explicit search skip/completion reasons and only runs grouped CV/grid search when every train fold can preserve full label coverage
- NOTE: training reports/runtime export metrics now include per-split accuracy + macro-F1 plus per-label precision/recall/F1/support and confusion-matrix payloads
- NOTE: live deployment now uses explicit promotion with bundle validation + backup
- NOTE: grouped search runs on train users only; skips cleanly if train split has <2 users
- NOTE: runtime now exposes `stable` + `eligible`; `eligible` is the future action-safe signal
- NOTE: `ModeRouter` now owns auth state transitions; runtime loop delegates auth policy
- NOTE: auth now uses a strict runtime auth-set subset built from configured digits plus mode-local aliases (`PEACE_SIGN -> TWO`, `OPEN_PALM -> FIVE` when `FIVE` is configured) and is implemented as a keypad-style committed buffer; pauses/no-hand frames do not erase entered digits
- NOTE: auth verification now happens only on explicit `BRAVO` submit against the committed buffer; `FIST` clears the buffer, `THUMBS_DOWN` pops one committed digit, and wrong submit triggers failure/lockout fail-safe behavior
- NOTE: sleep->auth wake and relock semantics now live in `ModeRouter`
- NOTE: presentation mode is now app-gated through `WindowWatch` and router permission sync
- NOTE: general-mode dry run now includes a stateful `PINCH_INDEX` primary interaction path using abstract cursor-space movement; cursor pose remains decoupled/provisional
- NOTE: `FIST` clutch now owns General Mode above primary interaction, cancels lower-priority primary state, and uses the same abstract cursor-space seam
- NOTE: `PINCH_MIDDLE` right-click now runs as a sibling secondary controller on the same seam, below clutch/primary ownership and emitting only on valid release
- NOTE: `SHAKA` scroll mode now owns General Mode beneath clutch, uses edge-triggered toggle + axis lock on the same seam, and suppresses primary/secondary progression while active
- NOTE: `CLOSED_PALM` is tuned as a compact flat-hand pose with fingers together and a compact thumb; detection now explicitly requires compact spread so it stays distinct from `OPEN_PALM/FIVE`
- NOTE: cursor preview now exists as the lowest-priority General Mode owner; cursor pose policy is isolated and still provisional (`L` by default)
- NOTE: safe OS execution now uses an explicit `dry_run`/`live` profile plus per-subsystem toggles; invalid or incomplete live config fails safe to no OS action
- NOTE: runtime execution now passes through `ExecutionSafetyGate`, which suppresses unsafe live cursor/primary/secondary/scroll output on hand loss or feature instability and taints unstable primary clicks
- NOTE: presentation mode now resolves `POINT_RIGHT/POINT_LEFT/OPEN_PALM/PEACE_SIGN` through a dedicated policy and uses the same safety + execution seams as General Mode
- NOTE: presentation mode is now explicitly scoped as playback-only for prepared slide decks; editing/settings actions remain out of scope until a separate policy is approved
- NOTE: presentation runtime now uses a local one-shot interpreter so held playback gestures do not auto-repeat; `OPEN_PALM` start and `PEACE_SIGN` exit require extra confirmation frames beyond navigation
- NOTE: `SHAKA` now exists as a canonical ops/unified label with conservative rule fallback detection, is defined without palm/back orientation assumptions, and is the default scroll-mode toggle gesture
- NOTE: `WindowWatch` now performs conservative foreground-app detection for PowerPoint, presentation-like browser tabs, and common PDF viewers; ambiguous context fails safe to no presentation action
- NOTE: `OPEN_PALM -> PRESENT_START` and `PEACE_SIGN -> PRESENT_EXIT` are now implemented as localized provisional presentation semantics because `FIST` is already overloaded elsewhere and `CLOSED_PALM` is considered too confusable for this repo
- NOTE: runtime/operator exit now uses an explicit lifecycle seam; manual `ESC/Q` exit and localized `THUMBS_DOWN` gesture exit both neutralize live/controller ownership before `EXITING`
- NOTE: centralized operator override policy now layers execution-profile and presentation-routing overrides onto the existing executor/router seams; invalid overrides fail safe to dry-run/auto-routing behavior
- BUG: validation not auto-run after recording save
- BUG: FPS target not yet met
- RISK: beam-search split planning is deterministic and fail-safe, but still heuristic for large user sets; broader real-data validation is still needed before calling the planner “finished”
- RISK: presentation context detection is intentionally conservative; unsupported apps/titles will fail safe to General Mode routing or no action

## NEXT STEP
1. Phase 7 focus: FPS, latency, and end-to-end runtime polish
2. Data-blocked path: train/export/promote first real runtime bundle
3. Optional future operator tooling: expose override policy through explicit UI/CLI without changing the runtime seams

## AGENT RULES
- Read: `PROJECT_TRACKER.md` -> `ARCHITECTURE_DECISIONS.md` -> `INTERFACES_AND_CONTRACTS.md` -> `OPEN_ISSUES.md` -> `SESSION_LOG.md`
- Respect locked rules; mark uncertain facts as `ASSUMED`/`UNVERIFIED`
- After any code change: update tracker files immediately
