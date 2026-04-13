Virtual Gesture Mouse - Phase 4 Dataset Collection Protocol

## Purpose

This protocol defines how to collect a clean, validation-ready Phase 4 dataset from the existing recorder in `tools/record_gestures.py`.

Primary objective:
- produce correctly labeled, schema-consistent, high-quality feature-vector sessions for later validation, filtering, training, and evaluation

Secondary objective:
- make collection simple enough that multiple operators can follow the same process without corrupting the dataset

This protocol is for data collection only.
It does not change the feature schema, the recorder save format, or the runtime gesture logic.

---

## Current Trainable Label Contract

The validation pipeline currently accepts the union of the auth and ops gesture sets:

1. `BRAVO`
2. `CLOSED_PALM`
3. `FIST`
4. `FIVE`
5. `FOUR`
6. `L`
7. `ONE`
8. `OPEN_PALM`
9. `PEACE_SIGN`
10. `PINCH_IM`
11. `PINCH_IMRP`
12. `PINCH_INDEX`
13. `PINCH_MIDDLE`
14. `PINCH_PINKY`
15. `PINCH_RING`
16. `POINT_LEFT`
17. `POINT_RIGHT`
18. `THREE`
19. `THUMBS_DOWN`
20. `TWO`

This 20-label inventory is the current trainable label contract for dataset validation.

---

## Trainable Gesture Sets

### Auth Set

Auth labels:
- `FIST`
- `CLOSED_PALM`
- `BRAVO`
- `THUMBS_DOWN`
- `ONE`
- `TWO`
- `THREE`
- `FOUR`
- `FIVE`

Use this set when collecting data specifically for authentication-sequence training or evaluation.

### Ops Set

Ops labels:
- `PINCH_IMRP`
- `PINCH_IM`
- `PINCH_INDEX`
- `PINCH_MIDDLE`
- `PINCH_RING`
- `PINCH_PINKY`
- `FIST`
- `CLOSED_PALM`
- `OPEN_PALM`
- `PEACE_SIGN`
- `L`
- `BRAVO`
- `THUMBS_DOWN`
- `POINT_RIGHT`
- `POINT_LEFT`

Use this set when collecting data for general interaction, control, and runtime gesture recognition.

### Presentation Set

Presentation labels:
- `POINT_RIGHT`
- `POINT_LEFT`
- `OPEN_PALM`
- `FIST`

This set is already fully covered by the ops set.
Do not create separate presentation-only labels in the dataset.

### Unified Set

If you want one dataset artifact that covers all currently trainable labels, collect the full 20-label union.

---

## Recommended Collection Strategy

Choose one of these collection scopes before recording:

1. Auth-only dataset:
- collect the 9 auth labels
- best when the immediate goal is ordered auth-sequence training

2. Ops-only dataset:
- collect the 15 ops labels
- best when the immediate goal is runtime control recognition

3. Unified dataset:
- collect the full 20-label union
- best when the goal is one reusable training pool for all current gesture categories

Recommended default:
- collect the unified dataset unless time is severely limited

Reason:
- the validator already accepts the unified label inventory
- overlapping labels such as `FIST`, `CLOSED_PALM`, `BRAVO`, and `THUMBS_DOWN` do not need to be recollected separately for each set
- presentation labels add no new classes beyond ops

---

## Minimum Dataset Targets

Collect at least:
- `2` users
- `2` backgrounds
- `60` accepted samples per label per session

Required minimum matrix:
- users: `U01`, `U02`
- backgrounds:
- `plain`
- `cluttered`
- `1` session per label per user per background

Minimum accepted sample totals by scope:

Auth-only:
- `9 labels x 2 users x 2 backgrounds x 60 = 2160 accepted samples`

Ops-only:
- `15 labels x 2 users x 2 backgrounds x 60 = 3600 accepted samples`

Unified:
- `20 labels x 2 users x 2 backgrounds x 60 = 4800 accepted samples`

Recommended extension after the minimum is complete:
- add `U03`
- repeat both backgrounds
- increase pinch-family sessions to `80` accepted samples each

---

## Session Rules

Each saved session must contain:
- exactly one gesture label
- exactly one user id
- exactly one background condition
- one consistent collection context

Never mix gesture classes inside one session.

Never save a session if:
- the operator repeatedly drifted into another gesture
- the gesture was not stable enough to hold
- the rejection count suggests poor capture quality for most of the session

Recommended discard rule:
- discard and restart a session if more than `40%` of attempted captures are rejected
- discard and restart if the operator visibly changed to a different gesture for part of the session

---

## Required Capture Metadata

For every session, provide these metadata fields through `--capture-context`:
- `background=plain` or `background=cluttered`
- `lighting=bright` or `lighting=mixed`
- `round=phase4_v2`

Optional but useful:
- `environment=dorm`
- `environment=lab`
- `camera=laptop_cam`
- `scope=auth`
- `scope=ops`
- `scope=unified`

Example:
```powershell
.\.venv\Scripts\python.exe tools\record_gestures.py --gesture-label FIST --user-id U01 --max-samples 60 --capture-context background=plain --capture-context lighting=bright --capture-context round=phase4_v2 --capture-context scope=unified
```

When both `scope` and `round` are present, the recorder now runs the readiness check automatically before opening the camera.
If either field is missing, the preflight is skipped so ad hoc recording still works normally.

---

## Recording Procedure

For each session:

1. Confirm the target label before opening the recorder.
2. Launch the recorder with the correct label, user id, target sample count, and capture-context metadata.
3. Keep only the intended gesture in the frame for the whole session.
4. Use the recorder HUD to verify:
- correct label
- correct user id
- correct mode
- gate reads `Ready to capture` before expecting acceptance
5. Use `AUTO` mode for stable gestures.
6. Use `MANUAL` mode for pinch-family labels if auto capture becomes noisy.
7. Watch accepted and rejected counters during the session.
8. If the session quality is poor, discard it and restart instead of saving weak data.
9. At the save confirmation screen, confirm the accepted count, rejected count, duration, and output path.
10. Save only if the session is clean.

---

## Recommended Per-Session Gesture Variation

Within the same gesture label, intentionally vary pose slightly while keeping the gesture class unchanged.

For each saved session, cover all three blocks:

1. Neutral block
- centered in frame
- natural distance from camera
- stable pose

2. Rotation block
- slight left rotation
- slight right rotation
- small wrist angle changes

3. Distance and position block
- slightly closer
- slightly farther
- slightly higher or lower in frame

Do not exaggerate the gesture so much that it becomes a different class.

For directional and thumb-led gestures, also vary:
- slight inward wrist roll
- slight outward wrist roll
- small changes in thumb spread while staying in the same class

For pinch-family gestures, also vary:
- pinch tightness
- finger spacing of the non-pinching fingers
- slight lateral repositioning in frame

---

## Recommended Capture Mode By Label

Use `AUTO` mode by default for:
- `BRAVO`
- `CLOSED_PALM`
- `FIST`
- `FIVE`
- `FOUR`
- `L`
- `ONE`
- `OPEN_PALM`
- `PEACE_SIGN`
- `POINT_LEFT`
- `POINT_RIGHT`
- `THREE`
- `THUMBS_DOWN`
- `TWO`

Prefer `MANUAL` mode when needed for:
- `PINCH_IM`
- `PINCH_IMRP`
- `PINCH_INDEX`
- `PINCH_MIDDLE`
- `PINCH_PINKY`
- `PINCH_RING`

If a pinch session shows too many rejections, switch to manual mode and capture only when the gate is stable.

---

## Recommended Collection Order

If you are collecting the unified dataset, use this order for every user:

1. `FIST`
2. `CLOSED_PALM`
3. `BRAVO`
4. `THUMBS_DOWN`
5. `ONE`
6. `TWO`
7. `THREE`
8. `FOUR`
9. `FIVE`
10. `OPEN_PALM`
11. `PEACE_SIGN`
12. `L`
13. `POINT_RIGHT`
14. `POINT_LEFT`
15. `PINCH_INDEX`
16. `PINCH_MIDDLE`
17. `PINCH_RING`
18. `PINCH_PINKY`
19. `PINCH_IM`
20. `PINCH_IMRP`

Complete all `plain` sessions first, then all `cluttered` sessions.

This reduces context switching and makes collection tracking simpler.

If you are collecting only one scope:
- auth-only: use the auth-set order shown above through `FIVE`
- ops-only: start from `FIST` and continue through `PINCH_IMRP`

---

## Command Templates

Plain background session:
```powershell
.\.venv\Scripts\python.exe tools\record_gestures.py --gesture-label FIST --user-id U01 --max-samples 60 --capture-context background=plain --capture-context lighting=bright --capture-context round=phase4_v2 --capture-context scope=unified
```

Cluttered background session:
```powershell
.\.venv\Scripts\python.exe tools\record_gestures.py --gesture-label FIST --user-id U01 --max-samples 60 --capture-context background=cluttered --capture-context lighting=mixed --capture-context round=phase4_v2 --capture-context scope=unified
```

Pinch session using manual mode after launch:
```powershell
.\.venv\Scripts\python.exe tools\record_gestures.py --gesture-label PINCH_INDEX --user-id U01 --max-samples 60 --capture-context background=plain --capture-context lighting=bright --capture-context round=phase4_v2 --capture-context scope=ops
```

After the window opens:
- press `M` if you want manual mode
- press `Space` to arm manual capture
- press `C` only when the gate says the hand is ready

---

## Step-by-Step Guide for Recording Gestures (For Non-Technical Users)

This section is for people helping with recording.
You do not need any coding experience.
Just follow the steps in order.

### A. Before You Start

1. Use a laptop or computer with a working webcam.
2. Sit where your face and hand are easy to see on camera.
3. Make sure the room is bright enough.
   Good light means your hand looks clear, not dark or shadowy.
4. Choose a background for the session.
   A plain wall is good for one session.
   A normal room background is good for another session.
5. Keep the camera steady.
   Do not move the laptop around during recording.
6. Sit at a comfortable distance.
   Your hand should fit fully inside the camera view.
7. Raise the recording hand so the full hand is visible.
   Do not let fingers go off-screen.
8. Remove anything that blocks the hand.
   Avoid sleeves, bags, or objects covering your fingers.
9. Avoid these problems before you begin:
   - very dark rooms
   - strong light behind you
   - busy movement in the background
   - sitting too far from the camera

### B. Starting the Program

1. I will give you the exact command to start.
2. A command is just the line you paste or run to open the recorder for one gesture.
3. Example command for recording the `FIST` gesture:
   ```powershell
   .\.venv\Scripts\python.exe tools\record_gestures.py --gesture-label FIST --user-id U01 --max-samples 60 --capture-context background=plain --capture-context lighting=bright --capture-context round=phase4_v2 --capture-context scope=unified
   ```
4. What this command means in simple words:
   - record the gesture called `FIST`
   - record for user `U01`
   - stop after `60` accepted samples
   - mark this session as a `plain` background session
   - mark this session as `bright` lighting
5. You do not need to build this command yourself.
   I can send the exact one for each gesture and recording condition.
6. Run the command the way I show you.
7. When it starts, you will see a camera window.
8. You will also see helpful text on the screen, such as:
   - the gesture name
   - the user name
   - the sample count
   - messages about whether the hand is ready
9. You may also see dots and lines drawn on your hand.
   That is normal.
   It helps the system follow your hand.
10. Important:
    the hand-tracking dots and lines should usually be visible while recording.
    If they disappear, the system may not be seeing your hand clearly.
11. You do not need a separate live validator window showing `STABLE = gesture name` in order to record.
    For recording, the main thing is that the recorder can see your hand and says it is ready to capture.

### C. Recording a Gesture

1. Record only one gesture in one session.
   Example: if the session is for `FIST`, only do `FIST`.
2. Put your hand in the camera view before starting.
3. Keep the full hand visible the whole time.
4. Make the correct gesture and hold it clearly.
5. Press `SPACE` to start recording.
   Press `SPACE` again if you want to pause.
6. Before expecting good captures, quickly check:
   - your full hand is visible
   - the hand-tracking lines are showing
   - the recorder says the hand is ready
7. If the session is using automatic capture, keep the gesture steady and let the program collect samples.
8. Move a little between samples so the data is more useful.
   Try:
   - a small left turn
   - a small right turn
   - slightly closer to the camera
   - slightly farther from the camera
   - slightly higher or lower in the frame
9. Keep the gesture the same while making those small changes.
   Do not turn it into a different gesture.
10. Press `C` to capture one single sample when manual capture is being used.
11. Press `U` if you want to remove the last sample you just recorded.
12. Press `Q` or `ESC` when you are done.
   This opens the save-and-exit step.
13. Save and exit only if the session looks clean and correct.
14. For the next gesture, you will get a new command.
   Example: `FIST` and `BRAVO` must be recorded in separate sessions.

### D. Best Practices

1. Record the same gesture in more than one lighting condition.
   Example: one bright session and one mixed-light session.
2. Record with more than one background.
   Example: one plain background and one normal room background.
3. Make small natural changes in hand angle and hand position.
   This helps the system learn real-life variations.
4. Keep the gesture shape consistent.
   Small position changes are good.
   Changing the gesture itself is bad.
5. Keep your hand relaxed but clear.
   Do not force an unnatural pose unless asked.
6. If the screen shows many failed captures, pause and reset your hand position before continuing.

### E. Common Mistakes To Avoid

1. Do not change gesture halfway through the session.
2. Do not move too fast.
   Fast movement makes the hand hard to track.
3. Do not let the hand leave the camera view.
4. Do not cover part of the hand with clothing or another object.
5. Do not record in very poor lighting.
6. Do not stand so far away that the hand looks small.
7. Do not save a session if you know the gesture was wrong for much of the recording.

### F. Repeating for Other Gestures

1. Finish and save the current gesture first.
2. Close that session.
3. Get the new command for the next gesture label.
4. Start a new session using that new command.
5. Repeat the full process again from the start.
6. Remember:
   one session = one gesture only.
7. If you need to record 10 different gestures, you must do 10 separate sessions.

### Quick Key Guide

1. `SPACE` → start or pause recording
2. `C` → capture one sample
3. `U` → undo the last sample
4. `Q` or `ESC` → save and exit

---

## Naming And Tracking Convention

User ids:
- `U01`
- `U02`
- `U03` if available

Background tags:
- `plain`
- `cluttered`

Round tag:
- `phase4_v2`

Scope tags:
- `auth`
- `ops`
- `unified`

The recorder already names files automatically by:
- gesture label
- user id
- session id

Do not manually rename the files after saving.

Use the tracker file:
- `docs/dataset_collection_tracker.csv`

Generate or refresh the tracker from code before collection:
```powershell
.\.venv\Scripts\python.exe tools\generate_dataset_tracker.py --scope unified --round phase4_v2 --output docs\dataset_collection_tracker.csv
```

For a narrower collection round, replace `unified` with `auth` or `ops`.

Validate readiness before recording:
```powershell
.\.venv\Scripts\python.exe tools\check_collection_readiness.py --scope unified --round phase4_v2
```

You do not need to run that command manually for normal managed collection sessions if the recorder launch includes:
- `--capture-context scope=...`
- `--capture-context round=...`

The standalone command is still useful when:
- you want to sanity-check the round before a recording block starts
- you regenerated the tracker and want confirmation before opening the recorder
- you are debugging a readiness failure

For every saved or discarded session, update the corresponding tracker row with:
- `accepted_samples`
- `rejected_attempts`
- `status`
- `notes`
- `file_path` for saved sessions

Recommended status values:
- `not_started`
- `in_progress`
- `completed`
- `discarded`
- `redo`

---

## What Counts As Complete

Auth-only dataset complete:
- all 9 auth labels are collected
- at least 2 users are fully collected
- both required backgrounds are fully collected
- every required session reached its accepted target
- no obviously corrupted or mixed-label sessions remain

Ops-only dataset complete:
- all 15 ops labels are collected
- at least 2 users are fully collected
- both required backgrounds are fully collected
- every required session reached its accepted target
- no obviously corrupted or mixed-label sessions remain

Unified dataset complete:
- all 20 current trainable labels are collected
- at least 2 users are fully collected
- both required backgrounds are fully collected
- every required session reached its accepted target
- no obviously corrupted or mixed-label sessions remain

Note:
- older checklist text that mentions `10 gestures` is from an earlier milestone and is no longer the live label contract

---

## Immediate Next Action

Recommended starting point for the current unified round:
- user `U01`
- background `plain`
- label `FIST`
- target `60`
- round `phase4_v2`
- scope `unified`

Then continue through the fixed collection order before moving to the `cluttered` background.
