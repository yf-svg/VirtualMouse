# AI CHANGE POLICY

Mandatory rule set for this repo. This policy applies before any implementation.

## PURPOSE
- Prevent unapproved behavior drift.
- Keep user-approved contracts as the source of truth.
- Force explicit approval before changing defaults, semantics, or operating rules.

## MUST FOLLOW
1. No behavior-contract change without explicit user approval.
   Behavior contracts include gesture meanings, mode semantics, auth semantics, routing rules, safety behavior, execution defaults, and user-facing control mappings.

2. "Provisional" does not mean "free to change".
   Provisional areas may be analyzed, tested behind an override, or proposed for improvement, but the default behavior must not change without approval.

3. Docs and memory record decisions; they do not create them.
   Tracker, contracts, and session logs may only document approved contract changes, not normalize unapproved implementation drift.

4. If code and approved docs disagree, approved docs/user instruction win.
   The implementation must be restored or an approval request must be made before changing the documented contract.

5. No default gesture-mapping changes without approval.
   This includes cursor, click, scroll, auth, presentation, exit, and mode-entry/exit mappings.

6. No execution-default changes without approval.
   This includes dry-run/live defaults, override defaults, and default safety/execution enablement.

7. Experiments must be isolated.
   Alternative behavior must go behind an explicit override, experimental config, or test-only path. It must not silently replace the default.

8. Approval must be specific.
   Before any contract/default change, state:
   - current contract
   - proposed change
   - reason / constraint
   - tradeoff or risk
   Then wait for explicit approval.

9. Contract restoration beats extension.
   If drift is found, restore the approved contract before adding new behavior unless the user explicitly chooses a new contract.

10. Violations must be disclosed directly.
    If an unapproved contract change happens, identify what changed, why it was a breach, what files were affected, and how to restore compliance.

## WORKING RULE
- Within approved contracts: implement directly.
- Outside approved contracts: explain first, get consent, then change.
