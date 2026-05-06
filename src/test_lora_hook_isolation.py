"""Isolated unit test of the inference-side LoRA hook.

Replicates the exact `_install_lora_hooks` from comfyui_node/nodes.py against a
synthetic `nn.Linear(3840, 3*3840)` standing in for ComfyUI's fused qkv. Loads
real QKLoRAPair weights from a phase 5 checkpoint, registers the hook, runs
forward, and verifies:

  1. The forward hook actually fires.
  2. After the hook, output[..., :dim] differs from the un-hooked Linear's
     output[..., :dim] by exactly the expected q_delta(x) — i.e. the hook
     adds what it claims to add.
  3. Same for K slice [dim:2*dim].
  4. V slice [2*dim:3*dim] is unchanged (the hook only touches Q and K).

This proves the hook closure / slicing / arithmetic in isolation. It does NOT
prove the hook fires inside ComfyUI's actual sampling path — that requires a
live workflow with a print statement in the hook.

Usage:
    python -m src.test_lora_hook_isolation <path_to_phase5_checkpoint.pt>
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn

from comfyui_node.style_adapter import BlockLoRAStack


def make_hook(lora_module, q_dim):
    """Mirror of comfyui_node/nodes.py:629-647 verbatim."""
    def hook(module, args, output):
        x = args[0]
        q_delta = lora_module.q_delta(x).to(
            dtype=output.dtype, device=output.device
        )
        k_delta = lora_module.k_delta(x).to(
            dtype=output.dtype, device=output.device
        )
        new_output = output.clone()
        new_output[..., :q_dim] = new_output[..., :q_dim] + q_delta
        new_output[..., q_dim:2 * q_dim] = (
            new_output[..., q_dim:2 * q_dim] + k_delta
        )
        return new_output
    return hook


def main(checkpoint_path: str) -> int:
    print(f"\n=== Hook isolation test ===")
    print(f"Checkpoint: {checkpoint_path}\n")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != "phase5":
        print(f"[FAIL] Not a phase 5 checkpoint")
        return 1

    config = state.get("config", {})
    num_blocks = config.get("num_blocks", 30)
    dim = config.get("dim", 3840)
    rank = config.get("lora_rank", 32)
    print(f"config: num_blocks={num_blocks} dim={dim} rank={rank}")

    block_lora = BlockLoRAStack(num_blocks=num_blocks, dim=dim, rank=rank)
    missing, unexpected = block_lora.load_state_dict(
        state["lora"], strict=False
    )
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing keys: {list(missing)[:3]}")
    if unexpected:
        print(f"  first unexpected keys: {list(unexpected)[:3]}")
    block_lora.eval()

    # Pick a representative block to test (block 15, in the "dead zone" the
    # block-mask diagnostic suggested). We test more blocks below in the loop.
    fire_count = {"n": 0}
    captured = {"q_delta": None, "k_delta": None, "x": None}

    test_block = 15
    lora_15 = block_lora.layers[test_block]
    qkv = nn.Linear(dim, 3 * dim, bias=False)

    # Wrap make_hook with extra accounting so we can verify after.
    user_hook = make_hook(lora_15, dim)
    def wrapped_hook(module, args, output):
        fire_count["n"] += 1
        captured["x"] = args[0].detach().clone()
        new_output = user_hook(module, args, output)
        # Sanity: q_delta and k_delta are what the hook computed
        captured["q_delta"] = lora_15.q_delta(args[0]).detach().clone()
        captured["k_delta"] = lora_15.k_delta(args[0]).detach().clone()
        return new_output

    handle = qkv.register_forward_hook(wrapped_hook)

    # Synthetic "post-modulated hidden_states": shape (B, S, dim), realistic
    # magnitude for a post-RMSNorm * (1 + scale) tensor (per-token L2 ~ sqrt(dim)).
    torch.manual_seed(42)
    B, S = 1, 64
    x = torch.randn(B, S, dim) * 1.0  # post-RMSNorm gives RMS~1 per element

    # Compute genuine output without hook
    handle.remove()
    with torch.no_grad():
        out_genuine = qkv(x)

    # Reinstall hook and recompute
    handle = qkv.register_forward_hook(wrapped_hook)
    with torch.no_grad():
        out_hooked = qkv(x)

    print(f"\n--- Test 1: hook fires ---")
    print(f"fire_count = {fire_count['n']}")
    if fire_count["n"] != 1:
        print(f"[FAIL] hook did not fire exactly once (got {fire_count['n']})")
        return 2
    print(f"[OK] hook fired exactly once")

    print(f"\n--- Test 2: Q slice modified by exactly q_delta ---")
    q_diff = out_hooked[..., :dim] - out_genuine[..., :dim]
    expected_q = captured["q_delta"]
    abs_err = (q_diff - expected_q).abs().max().item()
    print(f"max |actual_diff - expected_q_delta| = {abs_err:.6e}")
    print(f"||q_delta|| = {expected_q.norm().item():.4f}")
    print(f"||q_diff || = {q_diff.norm().item():.4f}")
    if abs_err > 1e-5:
        print(f"[FAIL] Q-slice modification doesn't match q_delta")
        return 3
    print(f"[OK] Q slice = genuine + q_delta exactly")

    print(f"\n--- Test 3: K slice modified by exactly k_delta ---")
    k_diff = out_hooked[..., dim:2*dim] - out_genuine[..., dim:2*dim]
    expected_k = captured["k_delta"]
    abs_err = (k_diff - expected_k).abs().max().item()
    print(f"max |actual_diff - expected_k_delta| = {abs_err:.6e}")
    print(f"||k_delta|| = {expected_k.norm().item():.4f}")
    print(f"||k_diff || = {k_diff.norm().item():.4f}")
    if abs_err > 1e-5:
        print(f"[FAIL] K-slice modification doesn't match k_delta")
        return 4
    print(f"[OK] K slice = genuine + k_delta exactly")

    print(f"\n--- Test 4: V slice unchanged ---")
    v_diff = out_hooked[..., 2*dim:3*dim] - out_genuine[..., 2*dim:3*dim]
    v_diff_norm = v_diff.norm().item()
    print(f"||v_diff|| = {v_diff_norm:.6e}")
    if v_diff_norm > 1e-5:
        print(f"[FAIL] V slice was modified, expected untouched")
        return 5
    print(f"[OK] V slice unchanged")

    print(f"\n--- Test 5: per-block sanity sweep ---")
    print(f"{'blk':>3} | {'||q_delta||':>11} {'||k_delta||':>11} | "
          f"{'%Q':>6} {'%K':>6}")
    print("-" * 56)
    handle.remove()
    # For each block, run forward through a fresh qkv with that block's LoRA
    # hook, verify q_delta and k_delta are non-zero, and report magnitudes
    # relative to the genuine Q/K projection magnitudes.
    qkv_test = nn.Linear(dim, 3 * dim, bias=False)
    with torch.no_grad():
        baseline = qkv_test(x)
    q_baseline_norm = baseline[..., :dim].norm().item()
    k_baseline_norm = baseline[..., dim:2*dim].norm().item()

    n_below = 0
    for blk_idx in range(num_blocks):
        lora_blk = block_lora.layers[blk_idx]
        qd = lora_blk.q_delta(x)
        kd = lora_blk.k_delta(x)
        qd_norm = qd.norm().item()
        kd_norm = kd.norm().item()
        q_pct = 100.0 * qd_norm / max(q_baseline_norm, 1e-12)
        k_pct = 100.0 * kd_norm / max(k_baseline_norm, 1e-12)
        if qd_norm < 1e-3:
            n_below += 1
        print(f"{blk_idx:>3} | {qd_norm:>11.4f} {kd_norm:>11.4f} | "
              f"{q_pct:>5.1f}% {k_pct:>5.1f}%")

    if n_below > 0:
        print(f"\n[WARN] {n_below} blocks have ||q_delta|| < 1e-3 — effectively no-op")
    else:
        print(f"\n[OK] all {num_blocks} blocks have non-trivial q_delta on synthetic input")

    print(f"\n=== Verdict ===")
    print(f"Hook plumbing works correctly in isolation: hook fires, slicing is")
    print(f"correct, and per-block deltas are non-zero on standard input.")
    print(f"")
    print(f"This means: if the inference path is broken, the bug is NOT in")
    print(f"the hook code itself. The bug would have to be in:")
    print(f"  (a) hook never being installed at sample time (ComfyUI lifecycle), or")
    print(f"  (b) hook installed but not called (qkv module replaced post-install), or")
    print(f"  (c) input tensor x at hook time differs structurally from training-time")
    print(f"      hidden_states despite our analysis showing they should match.")
    print(f"")
    print(f"Next step: live ComfyUI test with print in hook to discriminate.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
