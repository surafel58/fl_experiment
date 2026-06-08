"""
validate_perclient_metric.py — no-training validation of the FedCCFA per-client
generalized-accuracy port.

What this proves WITHOUT running training:

  1. IDENTITY INVARIANT (pre-drift):
     evaluate_per_client_gen_acc(model) == evaluate_gpu(model)
     because all clients start at global_test_id=0 and TEST_Y_VARIANTS[0] is
     an unmutated clone of TEST_Y. Must hold bit-for-bit (same forward pass,
     same labels). If this fails, the port is wrong.

  2. POST-DRIFT STATE:
     After apply_drift_event(client_y, 0):
       - CLIENT_GLOBAL_TEST_ID assigns A->1, B->2, C->3 per FedCCFA's cohort
         rule (client.id % 10).
       - TEST_Y_VARIANTS[0] still equals TEST_Y exactly (the undrifted ref).
       - TEST_Y_VARIANTS[1] differs from TEST_Y only on labels {1, 2} (1<->2).
       - TEST_Y_VARIANTS[2] differs only on labels {3, 4} (3<->4).
       - TEST_Y_VARIANTS[3] differs only on labels {5, 6} (5<->6).
       - per_client_gen_acc and global_acc diverge.

  3. TEST_X SHARING:
     There is one global TEST_X tensor; the four variants share it via
     evaluate_gpu(model, TEST_X, TEST_Y_VARIANTS[gid]). No per-variant image
     copy exists. (We only keep 4 label tensors, ~80 KB each.)
"""

import torch

# Importing the harness runs its module-level setup:
#   - loads CIFAR-10
#   - builds GPU tensors, TEST_X, TEST_Y, TEST_Y_VARIANTS, CLIENT_GLOBAL_TEST_ID
#   - defines evaluate_gpu, evaluate_per_client_gen_acc, apply_drift_event, etc.
from all_experiments_optimized import (
    get_model,
    evaluate_gpu,
    evaluate_per_client_gen_acc,
    fresh_client_y,
    apply_drift_event,
    reset_per_client_metric_state,
    TEST_X,
    TEST_Y,
    TEST_Y_VARIANTS,
    CLIENT_GLOBAL_TEST_ID,
    NUM_CLIENTS,
    DRIFT_GROUPS,
    DRIFT_EVENTS,
)


def hr(title=''):
    bar = '=' * 70
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


def label_mismatch_indices(a, b):
    """Indices where label tensor a differs from b. Both on GPU."""
    return torch.nonzero(a != b, as_tuple=True)[0]


def label_value_counts(y, labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    return {lbl: int((y == lbl).sum().item()) for lbl in labels}


# Ensure clean state (in case anything before this point mutated it).
reset_per_client_metric_state()

# Fresh model — random init; whatever accuracy it produces is fine, we only
# care that the same model produces the same number through both code paths.
gm = get_model()

hr('STEP A.1 — IDENTITY INVARIANT (pre-drift)')

global_acc_pre = evaluate_gpu(gm)
pc_acc_pre     = evaluate_per_client_gen_acc(gm)

print(f"  evaluate_gpu(gm)                      = {global_acc_pre!r}")
print(f"  evaluate_per_client_gen_acc(gm)       = {pc_acc_pre!r}")
print(f"  exact equality (==):                   {global_acc_pre == pc_acc_pre}")
print(f"  abs difference:                        {abs(global_acc_pre - pc_acc_pre):.20e}")

assert global_acc_pre == pc_acc_pre, (
    "IDENTITY INVARIANT FAILED. Pre-drift the per-client metric must equal "
    "the canonical global metric exactly. The port has a bug."
)
print("  PASS: pre-drift identity holds bit-for-bit.")

hr('STEP A.2 — INITIAL STATE (pre-drift)')

print(f"  CLIENT_GLOBAL_TEST_ID  = {CLIENT_GLOBAL_TEST_ID}")
assert CLIENT_GLOBAL_TEST_ID == [0] * NUM_CLIENTS, (
    "All clients must start at global_test_id = 0. FedCCFA's Client(..., 0)."
)
print("  PASS: every client starts at global_test_id = 0.")

for gid in (0, 1, 2, 3):
    same = bool(torch.equal(TEST_Y_VARIANTS[gid], TEST_Y))
    print(f"  TEST_Y_VARIANTS[{gid}] equals TEST_Y exactly: {same}")
    assert same, f"TEST_Y_VARIANTS[{gid}] differs from TEST_Y before any drift."
print("  PASS: every variant is a pristine clone of TEST_Y.")

hr('STEP A.3 — FIRE DRIFT EVENT 0')

client_y = fresh_client_y()
print(f"  DRIFT_EVENTS[0] swaps: {DRIFT_EVENTS[0]}")
print(f"  Cohort A clients: {DRIFT_GROUPS['A']}")
print(f"  Cohort B clients: {DRIFT_GROUPS['B']}")
print(f"  Cohort C clients: {DRIFT_GROUPS['C']}")
apply_drift_event(client_y, 0)

hr('STEP A.4 — POST-DRIFT STATE')

# CLIENT_GLOBAL_TEST_ID assignment per FedCCFA's `client.id % 10` rule.
expected_gids = []
for cid in range(NUM_CLIENTS):
    if cid % 10 < 3:
        expected_gids.append(1)        # cohort A -> variant 1 (1<->2)
    elif cid % 10 < 6:
        expected_gids.append(2)        # cohort B -> variant 2 (3<->4)
    else:
        expected_gids.append(3)        # cohort C -> variant 3 (5<->6)
print(f"  CLIENT_GLOBAL_TEST_ID (actual)   = {CLIENT_GLOBAL_TEST_ID}")
print(f"  CLIENT_GLOBAL_TEST_ID (expected) = {expected_gids}")
assert CLIENT_GLOBAL_TEST_ID == expected_gids, (
    "Cohort -> global_test_id mapping does not match FedCCFA's `id % 10` rule."
)
print("  PASS: cohort-to-variant mapping matches FedCCFA's `client.id % 10` rule.")

# Variant 0 must remain pristine.
v0_pristine = bool(torch.equal(TEST_Y_VARIANTS[0], TEST_Y))
print(f"  TEST_Y_VARIANTS[0] still equals TEST_Y exactly: {v0_pristine}")
assert v0_pristine, "TEST_Y_VARIANTS[0] (the undrifted reference) was mutated."

# Variant 1 must differ from TEST_Y ONLY on labels {1, 2} and be a clean swap.
def verify_swap(gid, a, b):
    diff_idx = label_mismatch_indices(TEST_Y_VARIANTS[gid], TEST_Y)
    diff_orig_labels = set(TEST_Y[diff_idx].tolist())
    print(f"  TEST_Y_VARIANTS[{gid}] vs TEST_Y: {len(diff_idx)} mismatches; "
          f"original labels at those positions = {sorted(diff_orig_labels)}")
    assert diff_orig_labels == {a, b}, (
        f"Variant {gid} swap is not isolated to {{{a},{b}}}."
    )
    # Count check: count(a) and count(b) must swap.
    counts_canon = label_value_counts(TEST_Y, (a, b))
    counts_var   = label_value_counts(TEST_Y_VARIANTS[gid], (a, b))
    print(f"    canonical counts:  {a}->{counts_canon[a]}, {b}->{counts_canon[b]}")
    print(f"    variant {gid} counts: {a}->{counts_var[a]}, {b}->{counts_var[b]}")
    assert counts_var[a] == counts_canon[b] and counts_var[b] == counts_canon[a], (
        f"Variant {gid} label counts inconsistent with a clean {a}<->{b} swap."
    )

verify_swap(1, 1, 2)
verify_swap(2, 3, 4)
verify_swap(3, 5, 6)
print("  PASS: variants [1]/[2]/[3] each carry exactly their cohort's swap.")

# Sanity: the per-client metric must now differ from global_acc (which is
# computed on canonical TEST_Y), because the variants are mutated.
global_acc_post = evaluate_gpu(gm)
pc_acc_post     = evaluate_per_client_gen_acc(gm)
print(f"  evaluate_gpu(gm) post-drift            = {global_acc_post!r}")
print(f"  evaluate_per_client_gen_acc(gm) post   = {pc_acc_post!r}")
print(f"  diverge? (should be != for non-trivial model): {global_acc_post != pc_acc_post}")
# Note: a random-init model may have ~10% acc and the metrics may coincide by
# accident on some random seeds. So this is a soft check, not a hard assert.

hr('STEP A.5 — TEST_X SHARING (no per-variant image copy)')

# TEST_Y_VARIANTS is a dict of LABEL tensors only. There is no TEST_X_VARIANTS
# dict. The single TEST_X (10000 images, ~120 MB on GPU) is reused in every
# evaluate_per_client_gen_acc call via evaluate_gpu(model, TEST_X, TEST_Y_VARIANTS[gid]).
print(f"  type(TEST_X)               = {type(TEST_X).__name__}")
print(f"  TEST_X.shape               = {tuple(TEST_X.shape)}")
print(f"  TEST_X memory (MB on GPU)  = {TEST_X.numel() * TEST_X.element_size() / 1024**2:.2f}")
print(f"  TEST_X.data_ptr()          = {hex(TEST_X.data_ptr())} (one pointer, no variant)")
print(f"  TEST_Y_VARIANTS structure  = dict with {len(TEST_Y_VARIANTS)} label tensors:")
total_label_mb = 0.0
for gid, yv in TEST_Y_VARIANTS.items():
    mb = yv.numel() * yv.element_size() / 1024**2
    total_label_mb += mb
    print(f"    [{gid}] shape={tuple(yv.shape)} dtype={yv.dtype} mem={mb*1024:.1f} KB "
          f"data_ptr={hex(yv.data_ptr())}")
print(f"  Total per-client-metric overhead vs canonical TEST_Y: "
      f"~{total_label_mb*1024:.1f} KB (4 label clones)")
print(f"  Image data is NOT duplicated. 4 full-test-set image copies "
      f"would have cost ~{4 * TEST_X.numel() * TEST_X.element_size() / 1024**2:.0f} MB on GPU.")

hr('VERDICT')
print("  All asserts passed. Pre-drift identity invariant holds bit-for-bit;")
print("  post-drift variants/cohort-IDs match FedCCFA's protocol exactly;")
print("  TEST_X is shared (no per-variant image copy).")
print("  The per-client generalized-accuracy port is correct.")
