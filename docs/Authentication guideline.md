# Authentication Guideline

## Purpose

This document describes the current authentication behavior in the repo:

- which gestures are allowed during auth
- what role each auth gesture has
- the default unlock flow
- timeout, reset, backtrack, and lockout behavior

Source of truth:

- [app/security/auth.py](c:/Users/win/Desktop/VirtualMouse/app/security/auth.py:8)
- [app/gestures/sets/auth_set.py](c:/Users/win/Desktop/VirtualMouse/app/gestures/sets/auth_set.py:1)
- [app/modes/router.py](c:/Users/win/Desktop/VirtualMouse/app/modes/router.py:64)

## Auth-Only Runtime Gesture Set

During locked/auth states, the runtime auth suite now allows only these gestures:

1. `ONE`
2. `TWO`
3. `THREE`
4. `BRAVO`
5. `FIST`
6. `THUMBS_DOWN`

Everything else is ignored by the auth suite during authentication.

That means gestures such as `CLOSED_PALM`, `OPEN_PALM`, `L`, `POINT_RIGHT`, `POINT_LEFT`, `PINCH_*`, and other non-auth gestures are not recognized as auth inputs.

## Gesture Roles

`ONE`
- Auth digit step 1 in the default sequence.

`TWO`
- Auth digit step 2 in the default sequence.

`THREE`
- Auth digit step 3 in the default sequence.

`BRAVO`
- Explicit approval gesture.
- Required after the full number sequence is entered.
- Auth does not complete until `BRAVO` is shown after `ONE -> TWO -> THREE`.

`FIST`
- Explicit reset gesture.
- Clears current auth progress immediately.

`THUMBS_DOWN`
- Explicit back gesture.
- Moves auth progress back by one step.
- If used after entering the full number sequence but before approval, it returns auth to the previous number step.

## Default Unlock Flow

The current default authentication flow is:

1. `ONE`
2. `TWO`
3. `THREE`
4. `BRAVO`

Important:

- `ONE..THREE` enter the auth sequence.
- `BRAVO` is the final approval step.
- The app does not unlock immediately after `THREE`; it waits for `BRAVO`.

## Runtime Configuration

Current defaults from [app/security/auth.py](c:/Users/win/Desktop/VirtualMouse/app/security/auth.py:8):

- `sequence = ("ONE", "TWO", "THREE")`
- `approve_gestures = ("BRAVO",)`
- `reset_gestures = ("FIST",)`
- `back_gestures = ("THUMBS_DOWN",)`
- `step_timeout_s = 4.0`
- `max_failures = 5`
- `cooldown_s = 10.0`

What this means:

- Each expected auth step must arrive within 4 seconds of the previous accepted step.
- `FIST` resets the whole attempt.
- `THUMBS_DOWN` moves back one step instead of failing the attempt.
- 5 failed attempts trigger lockout.
- Lockout lasts 10 seconds.

## Status Meanings

`idle`
- No auth progress yet.

`started`
- The first correct number step was accepted.

`progress`
- Auth is in progress.
- This includes the state after all numbers are entered and the system is waiting for `BRAVO`.

`step_back`
- `THUMBS_DOWN` moved progress back one step.

`success`
- `BRAVO` approved the fully entered sequence.
- The router can transition into active operation.

`reset_cancel`
- `FIST` reset the auth attempt.

`reset_wrong`
- A recognized auth gesture was used at the wrong time or in the wrong order.

`reset_timeout`
- The next expected step did not arrive before timeout.

`locked_out`
- Too many failures occurred.
- Auth is temporarily blocked until cooldown expires.

## Fail-Safe Behavior

Wrong recognized auth gesture:
- Resets progress and counts as a failure.

Timeout:
- Resets progress and counts as a failure.

Reset gesture:
- Clears progress without ambiguity.

Back gesture:
- Moves one step backward without counting as a failure.

Non-auth gesture during auth:
- Ignored by the auth suite instead of being treated as a valid auth candidate.

## Practical Example

If you want to unlock with the current defaults:

1. Show `ONE`
2. Show `TWO`
3. Show `THREE`
4. Show `BRAVO`

If you need to correct yourself:

- Show `THUMBS_DOWN` to go back one step
- Show `FIST` to reset the whole sequence

If you fail too many times:

- The app locks auth for 10 seconds
- After cooldown, auth can start again normally
