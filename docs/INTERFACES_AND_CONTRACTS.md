# INTERFACES + CONTRACTS

## CORE INTERFACES

### `HandTracker.detect(frame_bgr)`
- in: BGR frame
- out: `DetectedHand(landmarks, handedness)` | `None`
- invariant: MediaPipe-style 21 landmarks; single-hand path

### `assess_hand_input_quality(landmarks)`
- out: `HandInputQuality`
- invariant: rejects weak/partial/invalid hands before ML/rules

### `extract_feature_vector(landmarks)`
- out: `FeatureVector(values, schema_version)`
- invariant: schema = `phase3.v2`; fixed dimension

### `GestureEngine.process(hand|None)`
- out: `EngineOut(snapshot, candidates, decision, temporal, feature_vector, feature_temporal, quality)`
- invariant: validation-first; never raw label only

### `SVMClassifier.predict(feature_vector|None)`
- out: `ClassifierPrediction(label, confidence, accepted, reason, model_path)`
- invariant:
  - missing model must not crash runtime
  - runtime-approved bundle kind = `runtime_model_bundle`
  - `training_candidate` must be rejected at load time

### `GestureSuite.detect(hand)`
- out: `GestureSuiteOut(chosen, stable, eligible, candidates, reason, down, up, source, confidence, rule_chosen, ml_chosen, ml_reason, feature_reason, hold_frames, gate_reason)`
- invariant:
  - ML accepted only if schema/dim/confidence/feature_reason are valid
  - brief ML confidence dips may retain current ML label via hysteresis
  - otherwise rules fallback
  - `stable` = confirmed label; `eligible` = hold-qualified label for future auth/actions
  - `down/up` are `eligible` edges, not raw chosen edges

### `GestureAuth.update(gesture_label|None, now=...)`
- out: `GestureAuthOut(authenticated, status, matched_steps, total_steps, expected_next, consumed_label, failed_attempts, max_failures, retry_after_s)`
- invariant: consumes `eligible` edge labels only; ordered sequence + timeout + cancel reset + temporary lockout after repeated failures

### `ModeRouter.route_auth_edge(gesture_label|None, now=...)`
- out: `RouteOut(state, suite_key, auth_status, auth_progress_text, auth_out)`
- invariant: owns lock/authenticate/activate-general transitions; runtime loop must delegate auth state policy here

### `ModeRouter.request_sleep()` / `wake_for_auth()` / `lock()`
- out: `RouteOut(...)`
- invariant: sleep is only entered from active states; wake from sleep returns to auth path; lock always clears auth progress

### `ModeRouter.sync_presentation_permission(allowed)`
- out: `RouteOut(...)`
- invariant: only `ACTIVE_GENERAL -> ACTIVE_PRESENTATION` on allow and `ACTIVE_PRESENTATION -> ACTIVE_GENERAL` on deny; locked/auth states ignore presentation permission

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
- invariant: cursor-space input is gesture-agnostic; primary interaction must not depend on a fixed cursor-pose label

### `PrimaryInteractionController.update(gesture_label|None, cursor_point|None, now=...)`
- out: `PrimaryInteractionOut(state, intent, owns_state, movement, cursor_point)`
- invariant:
  - consumes only gated `PINCH_INDEX` signals + abstract cursor-space movement
  - drag starts from movement threshold, never duration alone
  - first valid release enters pending click; second valid release inside window -> double click
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
  - `PINCH_IM` toggles scroll mode only on valid edges; hold must not retrigger
  - active scroll suppresses primary/secondary progression
  - axis locks only after dead-zone exit + dominant movement margin
  - axis resets only on explicit conditions: exit, pause reset, prolonged hand loss

### `CursorPolicy.evaluate(gesture_label|None)`
- out: `CursorPolicyDecision(eligible, gesture_label, reason, provisional)`
- invariant:
  - current cursor pose assumption stays localized here
  - changing cursor pose later must not require interaction-controller rewrites

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

### `resolve_presentation_action(gesture_label, context)`
- out: `PresentationModeOut(intent, context)`
- invariant:
  - presentation logic stays separate from General Mode
  - `POINT_RIGHT/POINT_LEFT/OPEN_PALM/FIST` are resolved only inside presentation policy
  - unsupported or ambiguous context must produce `NO_ACTION`

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
- invariant: no split leakage; validated data separate from raw recordings

### `train_svm_from_validated_dataset(dataset_path, ...)`
- in: validated dataset artifact only
- out: candidate model artifact + training report
- invariant:
  - requires `split_status == ok`
  - requires `phase3.v2` + fixed dimension
  - grouped CV search runs on `train` users only when feasible
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
- `app/modes/general.py` -> full General Mode dry-run stack is defined; cursor pose policy remains provisional but isolated
- `app/modes/presentation.py` -> presentation resolver now exists; `OPEN_PALM` start and `FIST` exit remain provisional/localized until explicitly re-approved
- `app/control/actions.py` -> dry-run contract only
- `app/control/execution.py` / `mouse.py` -> live General + Presentation execution seams exist; explicit `dry_run`/`live` policy + runtime safety gate keep them fail-safe by default

## MUST NOT SILENTLY CHANGE
- `phase3.v2` feature contract
- `BRAVO` naming
- auth-only numbers
- ML-first -> rules-fallback semantics
- one gesture -> one action
- ambiguity -> no action
- recorder = one gesture per session
- scroll/clutch/cursor semantics require explicit design approval
