# INTERFACES + CONTRACTS

## CHANGE CONTROL

- `docs/AI_CHANGE_POLICY.md` is mandatory before implementation work.
- Contract/default changes require explicit user approval before code or doc updates.
- Tracker/docs may record approved decisions only; they must not normalize unapproved drift.

## CORE INTERFACES

### `HandTracker.detect(frame_bgr)`
- in: BGR frame
- out: `DetectedHand(landmarks, handedness)` | `None`
- invariant:
  - MediaPipe-style 21 landmarks; single-hand path
  - handedness exposed to runtime is normalized to the user's physical hand unless the inference input is explicitly marked mirrored

### `assess_hand_input_quality(landmarks)`
- out: `HandInputQuality`
- invariant: rejects weak/partial/invalid hands before ML/rules

### `extract_feature_vector(landmarks)`
- out: `FeatureVector(values, schema_version)`
- invariant: schema = `phase3.v2`; fixed dimension

### `GestureEngine.process(hand|None)`
- out: `EngineOut(snapshot, raw_candidates, candidates, decision, temporal, feature_vector, feature_temporal, quality)`
- invariant: validation-first; never raw label only

### `SVMClassifier.predict(feature_vector|None)`
- out: `ClassifierPrediction(label, confidence, accepted, reason, model_path)`
- invariant:
  - missing model must not crash runtime
  - runtime-approved bundle kind = `runtime_model_bundle`
  - `training_candidate` must be rejected at load time

### `GestureSuite.detect(hand)`
- out: `GestureSuiteOut(chosen, stable, eligible, raw_candidates, candidates, reason, down, up, source, confidence, rule_chosen, ml_chosen, ml_reason, feature_reason, hold_frames, gate_reason)`
- invariant:
  - ML accepted only if schema/dim/confidence/feature_reason are valid
  - brief ML confidence dips may retain current ML label via hysteresis
  - otherwise rules fallback
  - generic suite behavior stays validation-first; the live ops runtime may opt into explicit priority resolution through the ops-set runtime policy so click/scroll/control gestures survive overlap with lower-value shape labels
  - `raw_candidates` preserves detector output before mode filtering for debug/overlay use
  - `stable` = confirmed label; `eligible` = hold-qualified label for future auth/actions
  - `down/up` are `eligible` edges, not raw chosen edges

### `GestureAuth.update(gesture_label|None, now=...)`
- out: `GestureAuthOut(authenticated, status, matched_steps, total_steps, expected_next, consumed_label, failed_attempts, max_failures, retry_after_s, committed_sequence, buffer_full)`
- invariant:
  - consumes discrete auth-enter events only, not raw live pose state
  - owns the committed auth buffer for keypad-style entry
  - no-hand / pause frames must not erase committed digits
  - `BRAVO` validates the committed buffer only when the buffer is full
  - `THUMBS_DOWN` removes only the last committed digit; `SHAKA` clears the buffer
  - wrong submit may reset the buffer and enter temporary lockout after repeated failures

### `AuthGestureInterpreter.update(suite_out, auth_state, hand_present=...)`
- out: `AuthRuntimeOut(detected_gesture, event_label)`
- invariant:
  - auth-local gesture aliasing belongs here, not in the global classifier labels
  - current locked aliases are `PEACE_SIGN -> TWO` and `OPEN_PALM -> FIVE` when `FIVE` exists in the configured auth sequence
  - detected labels shown to auth UI are normalized auth symbols, while non-auth gestures stay hidden
  - emitted auth events remain discrete and release-gated

### `ModeRouter.route_auth_edge(gesture_label|None, now=...)`
- out: `RouteOut(state, suite_key, auth_status, auth_progress_text, auth_out)`
- invariant:
  - owns lock/authenticate/activate-general transitions; runtime loop must delegate auth state policy here
  - successful auth enters `ACTIVE_GENERAL` immediately; there is no separate hidden “enter general mode” control path

### `ModeRouter.request_sleep()` / `wake_for_auth()` / `lock()`
- out: `RouteOut(...)`
- invariant: sleep is only entered from active states; wake from sleep returns to auth path; lock always clears auth progress

### `ModeRouter.request_exit(source=..., reason=...)`
- out: `RouteOut(...)`
- invariant: runtime/operator shutdown must mark `EXITING` only after neutralization; router owns the final state transition rather than ad hoc loop breaks

### `ModeRouter.sync_presentation_permission(allowed)`
- out: `RouteOut(...)`
- invariant:
  - only `ACTIVE_GENERAL -> ACTIVE_PRESENTATION` on allow and `ACTIVE_PRESENTATION -> ACTIVE_GENERAL` on deny; locked/auth states ignore presentation permission
  - current product rule is automatic presentation activation from confident foreground context (or override), not gesture-entered presentation mode

### `WindowWatch.presentation_context()`
- out: `PresentationContext(...)`
- invariant:
  - foreground-app detection lives here, not in runtime/controller logic
  - ambiguous/unsupported context must fail safe to `allowed=False`
  - current conservative support is limited to PowerPoint, presentation-like browser tabs, and common PDF viewers

### `map_gesture_to_action(gesture_label)`
- out: `ActionIntent(action_name, gesture_label, executable, reason, mode)`
- invariant: general-mode dry run must not invent live control semantics; unmapped gestures stay non-executable

### `format_action_intent(intent)`
- out: overlay-friendly dry-run text
- invariant: formatting only; no OS side effects

### `cursor_point_from_landmarks(landmarks)`
- out: abstract normalized `CursorPoint(x, y)` | `None`
- invariant:
  - cursor-space input is gesture-agnostic; primary interaction must not depend on a fixed cursor-pose label
  - current runtime uses a palm-centered cursor anchor rather than fingertip anchoring so pinch articulation does not masquerade as drag/click movement
  - current runtime mirrors cursor-space x by default so live cursor motion matches the mirrored selfie preview

### `PrimaryInteractionController.update(gesture_label|None, cursor_point|None, now=...)`
- out: `PrimaryInteractionOut(state, intent, owns_state, movement, cursor_point)`
- invariant:
  - consumes only gated `PINCH_INDEX` signals + abstract cursor-space movement
  - drag starts from movement threshold, never duration alone
  - default fallback-mouse behavior emits single click immediately on a valid release
  - delayed pending/double-click semantics are optional policy, not the default live-click path
  - hand loss must fail safe; no ghost click/drag
  - current cursor-space point is exposed so live drag can reuse dry-run state without bypassing controllers

### `ClutchController.update(gesture_label|None, cursor_point|None, now=...)`
- out: `ClutchOut(state, intent, owns_state)`
- invariant:
  - consumes only gated `FIST`
  - owns General Mode above primary interaction
  - clutch activation cancels lower-priority primary state
  - release frame suppresses same-frame re-entry for safer no-jump behavior

### `SecondaryInteractionController.update(gesture_label|None, cursor_point|None, now=...)`
- out: `SecondaryInteractionOut(state, intent, owns_state, movement)`
- invariant:
  - consumes only gated `PINCH_MIDDLE` + abstract cursor-space movement
  - emits right click only on valid release, never pinch start
  - movement beyond tolerance cancels the candidate
  - hand loss must fail safe; no ghost right click

### `ScrollModeController.update(gesture_label|None, cursor_point|None, now=...)`
- out: `ScrollOut(state, axis, intent, owns_state, movement)`
- invariant:
  - the configured scroll-toggle gesture (current default: `SHAKA`) toggles scroll mode only on valid edges; hold must not retrigger
  - active scroll suppresses primary/secondary progression
  - axis locks only after dead-zone exit + dominant movement margin
  - axis resets only on explicit conditions: exit, pause reset, prolonged hand loss

### `CursorPolicy.evaluate(gesture_label|None)`
- out: `CursorPolicyDecision(eligible, gesture_label, reason, provisional)`
- invariant:
  - current cursor pose assumption stays localized here
  - current default cursor-ownership pose is `CLOSED_PALM`; changing it later must remain a policy/config decision, not a controller rewrite
  - changing cursor pose later must not require interaction-controller rewrites

### `resolve_operator_override_policy(...)`
- out: `ResolvedOperatorOverridePolicy`
- invariant:
  - overrides must remain centralized and fail-safe
  - current supported execution overrides include `cursor_test`, which enables live OS cursor output only for the cursor subsystem and leaves primary/secondary/scroll/presentation execution disabled
  - current default execution override is `fallback_live`, which enables the full live rule-fallback runtime through the existing executor/safety seams while keeping the underlying dry-run controllers unchanged

### `CursorPreviewController.update(..., higher_priority_owned=...)`
- out: `CursorPreviewOut(state, intent, owns_state, preview_point, movement, policy)`
- invariant:
  - cursor preview is lowest-priority General Mode owner
  - suppression must clear hand anchor but preserve preview position for no-jump re-entry
  - no OS cursor events in this layer

### `OSActionExecutor.apply_general_mode(general_out)`
- out: `ExecutionBatchReport(cursor, primary, secondary, scroll)`
- invariant:
  - consumes dry-run subsystem outputs only
  - must honor global execution gate before any OS side effect
  - executor may translate cursor/primary/secondary/scroll actions only; it must not infer gestures on its own
  - drag press/hold/release must stay faithful to primary dry-run outputs, with safe release on dry-run state exit or higher-priority override

### `ExecutionSafetyGate.evaluate(suite_out, general_out, hand_present)`
- out: `ExecutionSafetyDecision(...)`
- invariant:
  - runtime safety suppression belongs here, not in the backend
  - unstable features / hand loss may suppress live actions and taint pending primary clicks
  - dry-run controller semantics remain unchanged

### `ExecutionSafetyGate.evaluate_presentation(suite_out, presentation_out, hand_present)`
- out: `PresentationSafetyDecision(allow, reason)`
- invariant:
  - presentation actions must be suppressed on unstable features, hand loss, or untrusted context
  - no app/context heuristic may bypass dry-run action resolution

### `resolve_operator_override_policy(override_cfg, execution_cfg)`
- out: `ResolvedOperatorOverridePolicy(valid, reason, execution_override, routing_override, effective_execution)`
- invariant:
  - centralized override policy must layer onto execution config + router permission only
  - invalid/ambiguous override config must fail safe to conservative execution/routing
  - override policy may not bypass dry-run ownership or execution safety

### `OperatorLifecycleController.request_from_key(...)` / `request_from_suite_out(...)`
- out: `ExitRequest(...) | None`
- invariant:
  - manual exit is explicit and auditable
  - gesture exit must consume eligible edge output only and stay localized to active operator states
  - lifecycle exit policy stays separate from gesture/action controllers

### `neutralize_runtime_ownership(ctx, reason=...)`
- out: `RuntimeNeutralizationReport(...)`
- invariant:
  - shutdown/exit must release held mouse state, clear keyboard holds, and reset controller ownership before router exit
  - neutralization must be safe to call repeatedly

### `resolve_presentation_action(gesture_label, context)`
- out: `PresentationModeOut(intent, context)`
- invariant:
  - presentation logic stays separate from General Mode
  - Presentation Mode is playback-only for prepared slide decks; editing/settings actions must not be inferred here
  - `POINT_RIGHT/POINT_LEFT/OPEN_PALM` are resolved only inside presentation policy
  - consumes presentation-approved gesture events, not continuous held labels
  - unsupported or ambiguous context must produce `NO_ACTION`

### `PresentationGestureInterpreter.update(suite_out, hand_present)`
- out: `PresentationGestureSignal(gesture_label, event_label, active_frames, threshold_frames, reason)`
- invariant:
  - presentation runtime gating stays local to Presentation Mode
  - navigation emits one-shot events per held label; `OPEN_PALM` requires stricter confirmation than navigation
  - brief hand-loss/reacquire must not duplicate the same playback command
  - non-playback gestures remain invisible to presentation action routing

### `PresentationToolController.update(suite_out, hand_present, pointer_point)`
- out: `PresentationToolOut(state, intent, pointer_point, owns_presentation, stroke_active, stroke_capturing, selected_color_key, selected_pen_key, selected_size_key, panel_state, reason)`
- invariant:
  - presentation-local tool state is separate from slideshow playback routing
  - `L` toggles the presentation laser and `BRAVO` toggles draw mode
  - while draw mode is idle, `PEACE_SIGN` opens a right-justified rectangular tool tray; tray hits may consume `PINCH_INDEX` to change color, pen type, or size selection, and those hits must not start a draw stroke
  - the draw tray is hidden by default and must not block the slide corner during ordinary pointer travel; once opened, it closes again after the pointer leaves the tray bounds for a short grace window
  - draw-idle tray targeting uses a calmer presentation-local pointer path plus a brief pinch confirm so panel selection does not jump between options on normal hand jitter
  - tray targeting now flows through the shared pointer-filter seam (`deque`-backed recent history plus EMA output) rather than bespoke inline smoothing, so the selection path can be tuned independently without rewriting controller state logic
  - pointer motion slows locally inside the tray zone to make color/pen selection easier without slowing free drawing outside that zone
  - quick `FIST` release undoes one stroke, and sustained `FIST` hold clears annotations once; undo timing must tolerate recognition latency and short detected holds
  - `stroke_active` may remain true during draw release-grace to preserve controller ownership, but `stroke_capturing` must drop false immediately once pinch eligibility is lost so pen-up motion cannot append a tail segment
  - stroke capture should also stop on the earlier physical-intent seam (`chosen` dropping away from `PINCH_INDEX`) even before the later debounced `eligible` release completes, so release lag cannot draw a trailing tail line
  - stroke styling is captured at stroke start so later selector changes do not retroactively repaint existing annotations
  - the tray model is orthogonal: color, pen type, and size are independent selections rather than hardcoded combined presets

### `PresentationToolExecutor.apply(tool_out, context)`
- out: `PresentationToolExecutionReport(performed, reason, visible, laser_point, draw_point, stroke_count)`
- invariant:
  - presentation tool visuals are overlay-rendered here, not in playback routing or OS key execution
  - the draw tray stays hidden until explicitly opened in draw-idle mode, then animates in as a modern right-side overlay without affecting slideshow playback routing
  - annotation strokes are rendered with per-stroke color/pen style rather than a single mutable global style
  - stroke-point capture must follow `stroke_capturing`, not controller ownership alone, so release-grace frames finish the stroke without adding post-release points
  - active drawing should not inherit the heavier draw-idle/tray smoothing path; stroke capture uses its own lower-latency smoothing policy so inking stays more direct while idle/tray targeting can remain calmer
  - laser, draw-idle, and draw-stroking now use separate filter instances over a shared pointer-filter primitive (`deque` history + EMA output) so path tuning stays local to each interaction surface
  - draw visibility should tolerate short pointer dropouts near screen edges by briefly holding the last mapped draw point instead of blinking the pen out immediately
  - draw strokes should render as smooth brush paths rather than visibly jagged straight-segment chains
  - pen presets are semantically distinct (`pen`, `marker`, `highlighter`, `brush`, `quill`) and the overlay renderer should preserve that difference in both picker preview and stroke appearance
  - size presets are explicit tray selections and must scale stroke width independently from pen identity
  - presentation tool pointer mapping uses the dedicated presentation pointer anchor rather than the General Mode cursor anchor
  - presentation tool pointer mapping may also apply a presentation-local input-range remap so the slide reaches full height before the fingertip reaches the tracker edge
  - laser mode may briefly hold the last mapped point across short pointer dropouts so edge tracking does not flicker or teleport on transient recognition loss

### `MouseBackend.*`
- out: side effect only
- invariant:
  - backend boundary for OS cursor/click/drag/right-click/scroll execution
  - no controller may call OS APIs directly

### `OSActionExecutor.apply_presentation_mode(presentation_out, allow=..., suppress_reason=...)`
- out: `ExecutionReport(...)`
- invariant:
  - presentation OS actions must be translated only from presentation dry-run intents
  - executor may emit presentation keys only; it must not infer gestures or context
  - global execution policy and safety suppression must be honored before any key press

### `resolve_general_action(...)`
- out: `GeneralModeOut(intent, primary)`
- invariant: clutch priority > scroll mode > primary interaction > secondary interaction > cursor preview > explicit dry-run/no-op mapping; control ownership must stay conflict-free

### `validate_recording_files(paths, ...)`
- out: `ValidatedDataset`
- invariant:
  - no split leakage; validated data separate from raw recordings
  - `split_plan.status == ok` now implies every dataset label is represented in `train`
  - if leakage-safe three-way assignment cannot keep full train label coverage, validation must fail safe with non-`ok` split status
  - split planning may use exhaustive or deterministic beam-search assignment, but `split_plan.planner` / `assignment_score` must stay auditable in the artifact payload

### `train_svm_from_validated_dataset(dataset_path, ...)`
- in: validated dataset artifact only
- out: candidate model artifact + training report
- invariant:
  - requires `split_status == ok`
  - requires full train label coverage for every dataset label
  - requires `phase3.v2` + fixed dimension
  - grouped CV search runs on `train` users only when a group-aware plan preserves label coverage in every train fold
  - search skip/failure reasons must be explicit in the report/model payload; unexpected grouped-search failures must not silently degrade
  - training report metrics must stay structured enough for audit/export: split-level accuracy + macro-F1, per-label precision/recall/F1/support, and confusion-matrix counts
  - writes `training_candidate`; not runtime-approved directly

### `export_runtime_model_bundle(candidate_model_path, ...)`
- in: `training_candidate` artifact + optional training report
- out: exported `runtime_model_bundle`
- invariant:
  - carries model + schema + dim + labels + min_confidence + bundle version
  - output is the only model artifact type the runtime should trust by default

### `promote_runtime_model_bundle(source_model_path, ...)`
- in: exported `runtime_model_bundle`
- out: live model copy + optional backup
- invariant:
  - rejects non-runtime artifacts
  - validates bundle kind/version/schema/dim before promotion
  - backs up existing live model before replacement

### `tools/record_gestures.py`
- out: one-session-one-gesture JSON
- invariant: one label per session; readiness auto-check only when `scope` + `round` exist

## STUB CONTRACTS
- `app/modes/general.py` -> no action semantics defined yet
- `app/modes/general.py` -> full General Mode dry-run stack is defined; cursor pose policy remains provisional but isolated, with `CLOSED_PALM` as the current default ownership pose
- `app/modes/presentation.py` -> presentation resolver now exists; `OPEN_PALM` start remains provisional/localized, and presentation exit is currently owned by the localized `THUMBS_DOWN` lifecycle seam rather than the playback action map
- `app/modes/presentation_tools.py` / `app/control/presentation_tool_execution.py` -> presentation draw tools now include a `PEACE_SIGN`-summoned right-side tray for color and pen presets; the tray may intercept `PINCH_INDEX` locally in draw mode but must not alter slideshow playback semantics or General/Auth/Scroll mappings
- `app/control/actions.py` -> dry-run contract only
- `app/control/execution.py` / `mouse.py` -> live General + Presentation execution seams exist; explicit `dry_run`/`live` policy + runtime safety gate keep them fail-safe by default
- `app/control/mouse.py` -> Windows button/scroll injection uses `SendInput(...)` rather than legacy `mouse_event(...)`; executor must surface backend failures as explicit execution reasons instead of silently assuming success
- `app/lifecycle/runtime_loop.py` -> runtime emits structured terminal pipeline logs across detection, gesture selection, mode routing, General Mode controller ownership, safety, execution, and OS injection; diagnosis should prefer these real seam logs over overlay-only inference
- `app/control/execution_safety.py` -> feature-instability suppression must not kill clean release-driven clicks or active scroll output on the final transition frame alone; only genuinely tainted candidate ownership or hand loss should suppress those actions
- `app/control/execution_safety.py` -> when the runtime source is `rules` and `PINCH_INDEX` / `PINCH_MIDDLE` remain stable or eligible, click tainting must not be triggered by `feature_reason=unstable` alone; the fallback rules path is allowed to trust its own stable hold signal
- `app/control/execution_safety.py` -> prediction-gate `release_pending` frames that still carry stable/eligible `PINCH_INDEX` / `PINCH_MIDDLE` must remain trusted for click-taint decisions even if `source` has already fallen back to `none`; release debounce must not poison an otherwise clean click
- `app/control/execution_safety.py` -> the same `rules`-stable exception applies to the approved live cursor gesture, primary drag hold/start, and presentation playback gestures; feature-window instability alone must not suppress those actions while the rule-held gesture remains stably present
- `app/control/cursor_preview.py` / `app/control/execution.py` -> cursor preview activation must seed from the real current OS cursor position before the first active move, so entering General Mode preserves the visible pointer location instead of teleporting to the hand-normalized anchor
- `app/lifecycle/runtime_loop.py` -> manual `ESC/Q` exit is ignored until the first frame has actually been presented and the startup guard window has elapsed; this prevents spurious OpenCV startup key events from shutting the app down before it is usable
- `app/lifecycle/operator_lifecycle.py` -> manual exit is explicit; gesture exit is localized to `THUMBS_DOWN` eligible edges in active operator states and remains provisional
- `app/lifecycle/operator_lifecycle.py` -> `THUMBS_DOWN` on an eligible edge is mode-scoped: in `ACTIVE_PRESENTATION` it requests a presentation-off route change instead of process exit, and in `ACTIVE_GENERAL` it may exit the app only when higher-priority controllers are neutral so scroll/drag/clutch transitions cannot accidentally terminate the session
- `app/lifecycle/operator_lifecycle.py` -> after a presentation-off `THUMBS_DOWN`, gesture exit must stay latched until that same gesture fully clears; a still-held presentation exit gesture must not immediately cascade into General Mode app exit on the next frame
- `app/modes/router.py` -> manual presentation-off requests latch General Mode while presentation context remains allowed; auto-presentation may re-enter only after context first clears and then becomes allowed again
- `app/lifecycle/operator_policy.py` -> centralized execution/routing override policy exists; invalid overrides fail safe to conservative routing and non-live execution
- `tools/record_gestures.py` -> recorder is now rules-guided and capture-safe: target gesture labels must be canonical, default capture requires both quality pass and target-label eligibility, overwriting existing output requires explicit `--overwrite`, tracker sync happens automatically when full collection context is present unless `--no-tracker-sync` is used, compact raw-landmark sidecars are written by default, and optional accepted-sample snapshot sidecars must stay storage-conscious hand crops rather than full-frame dumps

## MUST NOT SILENTLY CHANGE
- `phase3.v2` feature contract
- `BRAVO` naming
- auth-only numbers
- ML-first -> rules-fallback semantics
- one gesture -> one action
- ambiguity -> no action
- recorder = one gesture per session
- scroll/clutch/cursor semantics require explicit design approval
