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
- Presentation subset: `POINT_RIGHT POINT_LEFT OPEN_PALM THUMBS_DOWN`
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
- NOTE: auth success transitions directly into `ACTIVE_GENERAL`; runtime/debug now surfaces this explicitly as `MODE:GENERAL_READY` so post-auth control readiness is observable instead of implicit
- NOTE: auth now uses a strict runtime auth-set subset built from configured digits plus mode-local aliases (`PEACE_SIGN -> TWO`, `OPEN_PALM -> FIVE` when `FIVE` is configured) and is implemented as a keypad-style committed buffer; pauses/no-hand frames do not erase entered digits
- NOTE: auth verification now happens only on explicit `BRAVO` submit against the committed buffer; `SHAKA` clears the buffer, `THUMBS_DOWN` pops one committed digit, and wrong submit triggers failure/lockout fail-safe behavior
- NOTE: sleep->auth wake and relock semantics now live in `ModeRouter`
- NOTE: presentation mode is now app-gated through `WindowWatch` and router permission sync
- NOTE: Presentation Mode activation is currently automatic, not gesture-entered: `ACTIVE_GENERAL -> ACTIVE_PRESENTATION` occurs only when foreground presentation context is confidently allowed (or an override forces it), and runtime/debug now surfaces this as `MODE:PRESENTATION_AUTO` or `MODE:PRESENTATION_FORCED`
- NOTE: general-mode dry run now includes a stateful `PINCH_INDEX` primary interaction path using abstract cursor-space movement; cursor pose remains decoupled/provisional
- NOTE: fallback live-click semantics now prioritize immediate `PINCH_INDEX` click on valid release; delayed pending/double-click behavior is optional and no longer the default virtual-mouse click path
- NOTE: `FIST` clutch now owns General Mode above primary interaction, cancels lower-priority primary state, and uses the same abstract cursor-space seam
- NOTE: `PINCH_MIDDLE` right-click now runs as a sibling secondary controller on the same seam, below clutch/primary ownership and emitting only on valid release
- NOTE: `SHAKA` scroll mode now owns General Mode beneath clutch, uses edge-triggered toggle + axis lock on the same seam, and suppresses primary/secondary progression while active
- NOTE: `CLOSED_PALM` is tuned as a compact flat-hand pose with fingers together and a modestly tucked thumb; `CLOSED_PALM` and `OPEN_PALM/FIVE` now have an intentional geometric dead zone between them so borderline flat hands fail safe to neither label instead of collapsing into `OPEN_PALM`
- NOTE: cursor preview now exists as the lowest-priority General Mode owner; cursor pose policy is isolated and still provisional, with `CLOSED_PALM` as the current default ownership pose
- NOTE: shared cursor-space input is now palm-centered and mirrors x by default, so live cursor motion matches the mirrored preview and pinch articulation no longer looks like cursor drag motion to the click controllers
- NOTE: General Mode cursor activation now seeds its preview position from the real current OS cursor before the first active move, so entering cursor control no longer teleports the pointer toward the hand-normalized position or makes it appear to vanish at an edge
- NOTE: safe OS execution now uses an explicit `dry_run`/`live` profile plus per-subsystem toggles; invalid or incomplete live config fails safe to no OS action
- NOTE: operator override now supports `cursor_test`, which enables live OS cursor movement only while keeping primary/secondary/scroll/presentation execution disabled; this is the safest supported path for rule-fallback virtual-mouse testing
- NOTE: the repo-default runtime now resolves through a centralized `fallback_live` operator override so the rule-fallback system comes up as a real live virtual mouse without bypassing the dry-run controller/safety architecture
- NOTE: runtime execution now passes through `ExecutionSafetyGate`, which suppresses unsafe live cursor/primary/secondary/scroll output on hand loss or feature instability and taints unstable primary clicks
- NOTE: Windows live mouse button/scroll injection now uses `SendInput(...)` with executor-visible backend error reporting, so click failures are diagnosable at the OS seam rather than disappearing silently behind gesture/controller success
- NOTE: runtime now emits structured terminal trace lines (`[DETECTION]`, `[GESTURE]`, `[MODE]`, `[GENERAL]`, `[SAFETY]`, `[EXEC]`, `[OS]`, `[OS_ERROR]`) so post-recognition failures can be proven at the real runtime seams instead of guessed from overlay text alone
- NOTE: `ExecutionSafetyGate` now distinguishes unstable transition frames from truly tainted controller state: clean release-driven clicks and active scroll output are allowed to execute even if the final frame is feature-unstable, while instability or hand loss during candidate ownership still taints/suppresses the resulting action
- NOTE: rule-held stable/eligible pinch candidates are now trusted over `feature_reason=unstable` for click tainting; the fallback rules path no longer poisons a click just because the feature window is noisy while `PINCH_INDEX`/`PINCH_MIDDLE` remain stably held
- NOTE: the same rule-held-stable exception now also covers live cursor motion, primary drag start/hold, and presentation playback actions; when the source is `rules` and the approved gesture remains stable/eligible, feature-window instability alone no longer suppresses that live action
- NOTE: prediction-gate `release_pending` frames now remain trusted for stable/eligible pinch holds even when `source=none`; release-driven primary/secondary clicks no longer get falsely tainted during the gate's final debounce frame
- NOTE: startup manual `ESC/Q` exit is now guarded until the first rendered frame plus a short startup grace, so OpenCV startup key noise cannot immediately tear down the app before the window becomes usable
- NOTE: General Mode overlay now exposes cursor policy + execution reasons separately (`CUR:... | CUREX:...`) so lack of cursor motion is diagnosable as policy rejection, ownership suppression, or dry-run/live config
- NOTE: live ops runtime now uses an explicit priority-enabled ops-suite policy so `PINCH_INDEX`/`PINCH_MIDDLE` survive overlap with lower-value shape labels instead of being nulled out as ambiguous; generic `GestureSuite()` remains validation-first for measurement/tests
- NOTE: General Mode overlay now also surfaces the active live-fallback control legend (`MOVE=CLOSED_PALM`, `PINCH_INDEX`, `PINCH_MIDDLE`, `SHAKA`, `FIST`) so rule-based mouse testing is explicit instead of hidden behind a provisional pose assumption
- NOTE: the dataset collection contract is now 21 labels, not 20; `SHAKA` is part of the canonical ops/unified training inventory, and the tracker/protocol template have been regenerated to the 84-row `U01/U02 x plain/cluttered x unified` matrix for `phase4_v2`
- NOTE: the recorder now defaults to a rules-only target-label confirmation guard: captures only succeed when hand quality passes and the intended label is action-eligible; per-sample payloads also record the recorder's guidance label state for later audit
- NOTE: the recorder now validates target labels, refuses to overwrite existing output files unless `--overwrite` is supplied, and auto-syncs the matching tracker row on save/discard when `round/scope/background/lighting` context is present
- NOTE: the recorder now writes a compact raw-landmark sidecar (`.landmarks.npz`) by default using compressed `float16` MediaPipe `x/y/z` coordinates aligned to accepted samples; optional `--save-snapshots` sidecars are stored as cropped low-resolution JPEG hand images to maximize ML/audit value per MB
- NOTE: presentation mode now resolves `POINT_RIGHT/POINT_LEFT/OPEN_PALM` through a dedicated playback policy and keeps `THUMBS_DOWN` as the separate lifecycle exit path
- NOTE: presentation mode is now explicitly scoped as playback-only for prepared slide decks; editing/settings actions remain out of scope until a separate policy is approved
- NOTE: presentation runtime now uses a local one-shot interpreter so held playback gestures do not auto-repeat; `OPEN_PALM` start requires extra confirmation frames beyond navigation
- NOTE: `SHAKA` now exists as a canonical ops/unified label with conservative rule fallback detection, is defined without palm/back orientation assumptions, and is the default scroll-mode toggle gesture
- NOTE: runtime handedness is now normalized to the user's physical hand from MediaPipe's mirrored/selfie assumption, and the debug overlay exposes raw detector candidates separately from mode-filtered candidates
- NOTE: `WindowWatch` now performs conservative foreground-app detection for PowerPoint, presentation-like browser tabs, and common PDF viewers; ambiguous context fails safe to no presentation action
- NOTE: `OPEN_PALM -> PRESENT_START` remains the only slideshow session-control semantic in the playback resolver; `THUMBS_DOWN` owns presentation exit through the lifecycle seam instead of the playback map
- NOTE: presentation tools now have a dedicated dry-run controller seam in `ACTIVE_PRESENTATION`; `L` toggles laser, `BRAVO` toggles draw mode, and `PEACE_SIGN` now opens a right-justified draw tray for style choices without changing General/Auth/Scroll behavior
- NOTE: the draw tray stays hidden by default, opens only on intentional `PEACE_SIGN` summon in draw-idle mode, and `PINCH_INDEX` on a tray option changes color, pen type, or size instead of starting a stroke
- NOTE: selector targeting is now calmer in draw-idle mode: the panel uses a smoothed local pointer path and a brief pinch confirm before changing color/pen, which reduces option-jumping on natural hand jitter
- NOTE: the presentation draw tray now exposes a fuller modern annotation rail: 6 color swatches, 5 pen identities (`pen` / `marker` / `highlighter` / `brush` / `quill`), and 5 explicit size presets so stroke width is selected independently from pen identity
- NOTE: the lower-right corner summon dot/beacon has been removed entirely; the draw tray no longer occupies or obscures the bottom-right slide area while idle
- NOTE: pointer motion now slows locally inside the tray zone to make color/pen picking easier, while free drawing outside the tray keeps the faster path
- NOTE: the open draw tray now closes automatically after the pointer leaves its bounds for a short grace window, which keeps dismissal lightweight without flicker
- NOTE: draw release-grace now preserves controller ownership without preserving stroke capture; pen-up motion must no longer append a final tail segment after `PINCH_INDEX` is released
- NOTE: active draw capture now also stops on the earlier `chosen`-label drop, not only on the later debounced `eligible` release, so post-release finger motion should no longer paint a trailing tail while the gesture gate is still unwinding
- NOTE: active drawing now uses its own faster adaptive smoothing path with short raw-point history instead of inheriting the calmer draw-idle/tray smoothing path, which should reduce perceived pen lag without making tray targeting twitchy
- NOTE: presentation tray targeting, laser motion, and active stroke capture now share a common pointer-filter seam built on short recent-point history plus EMA output; each path keeps its own filter instance so tuning remains local while implementation stays consistent
- NOTE: presentation draw strokes now preserve the color/pen style that was active when each stroke started; later panel changes do not retroactively repaint existing annotations
- NOTE: draw strokes are now rendered as smoother brush paths instead of obvious straight-segment chains, and pen styling is now modeled as `color + pen type + size` rather than a small fixed set of width-only presets
- NOTE: presentation draw erase timing is now latency-tolerant: a quick `FIST` release is accepted across short detected holds and natural release lag, while sustained `FIST` hold still clears annotations once at the dedicated clear threshold
- NOTE: presentation tools now use their own dedicated pointer anchor path (`index_tip` by default) instead of reusing the General Mode cursor anchor, which stabilizes the laser/draw cursor near slide edges and the bottom of the window
- NOTE: presentation tools now also apply a presentation-local pointer input-range remap, so the slide can reach full bottom height before the fingertip reaches the bottom of the tracking frame; this is meant to stop laser dropout in the lower slide region on real camera setups
- NOTE: laser rendering now briefly holds the last mapped point across short pointer dropouts so transient recognition loss does not make the dot disappear and reappear in a distant corner
- NOTE: draw cursor rendering now also briefly holds the last mapped point across short pointer dropouts, which should make edge travel feel less fragile instead of blinking the pen out immediately near screen boundaries
- NOTE: runtime/operator exit now uses an explicit lifecycle seam; manual `ESC/Q` exit and localized `THUMBS_DOWN` gesture exit both neutralize live/controller ownership before `EXITING`
- NOTE: `THUMBS_DOWN` now has mode-specific lifecycle behavior: in `ACTIVE_PRESENTATION` it turns Presentation Mode off, keeps auto-presentation latched until context clears, and also blocks General Mode gesture-exit rearm until the held thumbs-down gesture is released; in `ACTIVE_GENERAL` it exits the app only from a safe neutral state (not while scroll/clutch/primary/secondary still own control)
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
- Read: `PROJECT_TRACKER.md` -> `AI_CHANGE_POLICY.md` -> `ARCHITECTURE_DECISIONS.md` -> `INTERFACES_AND_CONTRACTS.md` -> `OPEN_ISSUES.md` -> `SESSION_LOG.md`
- Respect locked rules; mark uncertain facts as `ASSUMED`/`UNVERIFIED`
- After any code change: update tracker files immediately
- Before implementation, follow `AI_CHANGE_POLICY.md` as a mandatory gate for any contract/default/behavior change
