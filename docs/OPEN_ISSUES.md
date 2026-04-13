# OPEN ISSUES

## ISSUE-001 | registry precedence hidden
- sev: M
- status: OPEN
- files: `app/gestures/registry.py`
- symptom: `THUMBS_DOWN` + `FIST` return early -> overlap stats incomplete
- cause: short-circuit safety hack
- next: move full precedence to later stage
- block: non-blocking now; blocks clean ambiguity accounting

## ISSUE-002 | no trained runtime model
- sev: H
- status: OPEN
- files: `app/gestures/classifier.py`, `app/gestures/training.py`, `tools/train_svm.py`
- symptom: classifier reason = `model_unavailable`; runtime uses rules
- cause: export path now exists, but no real evaluated runtime bundle is deployed to `models/gesture_svm.joblib`
- next: train on real validated data, export runtime bundle, then test ML path live
- block: blocks true ML runtime

## ISSUE-003 | auth/mode/control stubs
- sev: H
- status: OPEN
- files: `app/security/auth.py`, `app/modes/*`, `app/control/*`
- symptom: no auth, no mode routing, no OS output
- cause: later phases not implemented
- next: build on top of unified `GestureSuiteOut`
- block: blocks user-visible app behavior

## ISSUE-004 | runtime loop still validator-style
- sev: M
- status: OPEN
- files: `app/lifecycle/runtime_loop.py`
- symptom: overlay-only loop; no auth/mode/action dispatch
- cause: runtime built as inspection tool first
- next: refactor loop around classifier seam + mode router
- block: blocks Phase 5/6

## ISSUE-005 | validation not mandatory after recording
- sev: M
- status: OPEN
- files: `tools/record_gestures.py`, `tools/validate_recordings.py`
- symptom: raw recordings can exist without immediate validation
- cause: readiness automation added before validation automation
- next: decide per-save vs batch validation hook
- block: non-blocking, workflow risk

## ISSUE-006 | FPS target unmet
- sev: M
- status: OPEN
- files: runtime/perception path
- symptom: last documented path ~19.8 FPS; target >=25
- cause: likely CPU path cost
- next: add repeatable benchmark before retuning
- block: non-blocking now; Phase 7 risk

## ISSUE-007 | scroll/clutch/cursor semantics undefined
- sev: M
- status: OPEN
- files: future `app/control/*`
- symptom: control behavior not approved/implemented
- cause: architecture moved ahead of action design
- next: explicit control spec before Phase 6
- block: blocks control implementation
