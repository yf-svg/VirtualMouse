# Authentication Guideline

## Purpose

This document describes the current authentication behavior in the repo:

- which gestures are allowed during auth
- what role each auth gesture has
- the default unlock flow
- keypad-style buffer, submit, reset, backtrack, and lockout behavior

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
7. `PEACE_SIGN`

Everything else is ignored by the auth suite during authentication.

That means gestures such as `CLOSED_PALM`, `L`, `POINT_RIGHT`, `POINT_LEFT`, `PINCH_*`, and other non-auth gestures are not recognized as auth inputs.

Mode-local aliases:

- `PEACE_SIGN` is accepted as auth step `TWO`
- `OPEN_PALM` is accepted as auth step `FIVE` only when `FIVE` is present in the configured auth sequence

These aliases are handled only inside the auth runtime path. They do not rename the canonical detector vocabulary globally.

## Gesture Roles

`ONE`
- Auth digit step 1 in the default sequence.

`TWO`
- Auth digit step 2 in the default sequence.
- Auth mode also accepts `PEACE_SIGN` as the physical pose for this step.

`THREE`
- Auth digit step 3 in the default sequence.

`BRAVO`
- Explicit approval gesture.
- Required after the full number sequence is entered.
- Validates the committed auth buffer.
- Auth does not complete until `BRAVO` is shown after `ONE -> TWO -> THREE`.

`FIST`
- Explicit reset gesture.
- Clears the committed auth buffer immediately.

`THUMBS_DOWN`
- Explicit back gesture.
- Removes only the most recently committed auth digit.
- If used after entering the full number sequence but before approval, it returns auth to the previous digit step.

## Default Unlock Flow

The current default authentication flow is:

1. `ONE`
2. `TWO`
3. `THREE`
4. `BRAVO`

Important:

- `ONE..THREE` are entered like keypad digits into a committed auth buffer.
- Entered digits stay latched even if the hand disappears or changes pose.
- `BRAVO` is the final approval step and validates the committed buffer.
- The app does not unlock immediately after `THREE`; it waits for `BRAVO`.
- `THUMBS_DOWN` edits the stored buffer; `FIST` clears it.

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

- Auth input is latched; the user may pause between digits without losing committed entries.
- `FIST` resets the whole buffer.
- `THUMBS_DOWN` removes one committed digit instead of failing the attempt.
- `BRAVO` validates the committed buffer only after the buffer is full.
- `step_timeout_s` remains in config for compatibility, but it no longer clears committed auth input during normal keypad-style entry.
- 5 failed attempts trigger lockout.
- Lockout lasts 10 seconds.

If a future auth sequence includes `FIVE`:

- the auth runtime will accept the canonical `OPEN_PALM` detector label as auth step `FIVE`
- this remapping stays localized to auth mode so training/runtime can keep `OPEN_PALM` as the canonical detector label

## Status Meanings

`idle`
- No auth progress yet.

`started`
- The first correct number step was accepted.

`progress`
- Auth is in progress.
- This means at least one committed digit is stored and the buffer is not full yet.

`ready_to_submit`
- All required digits are stored in the committed buffer.
- The system is waiting for `BRAVO`.

`step_back`
- `THUMBS_DOWN` moved progress back one step.

`success`
- `BRAVO` approved the fully entered sequence.
- The router can transition into active operation.

`reset_cancel`
- `FIST` reset the auth attempt.

`reset_wrong`
- The committed buffer failed validation when submitted with `BRAVO`.

`locked_out`
- Too many failures occurred.
- Auth is temporarily blocked until cooldown expires.

## Fail-Safe Behavior

Wrong recognized auth gesture:
- Does not erase committed digits by itself.

Reset gesture:
- Clears the committed buffer without ambiguity.

Back gesture:
- Removes one committed digit without counting as a failure.

Non-auth gesture during auth:
- Ignored by the auth suite instead of being treated as a valid auth candidate.

Pause / no hand:
- Does not erase committed digits.
- No new digit is committed until the user presents the next intentional auth gesture.

## Practical Example

If you want to unlock with the current defaults:

1. Show `ONE`
2. Relax or reposition if needed
3. Show `TWO`
4. Relax or reposition if needed
5. Show `THREE`
6. Show `BRAVO`

If you need to correct yourself:

- Show `THUMBS_DOWN` to go back one step
- Show `FIST` to reset the whole sequence

If you fail too many times:

- The app locks auth for 10 seconds
- After cooldown, auth can start again normally
