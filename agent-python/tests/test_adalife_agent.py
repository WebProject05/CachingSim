"""
Unit tests for AdaLife-Pro agent integration, unit-exact candidate features,
and yield-oracle expert action logic.
"""

import numpy as np
from pb import cache_env_pb2
from config import AgentConfig
from client.grpc_env_client import state_response_to_numpy
from algorithms.ddql import DDQLAgent


def _make_state_resp(total_files: int, mem_ratio: float, req_file: int):
    d = np.zeros(total_files, dtype=np.float32)
    y = np.full(total_files, 0.5, dtype=np.float32)
    z = np.full(total_files, 500.0, dtype=np.float32)  # 500 MiB = 0.5 GiB
    b = np.zeros(total_files, dtype=np.int32)
    b[0] = 1
    d[req_file] = 3.0
    d[0] = 10.0
    return cache_env_pb2.StateResponse(
        mem=mem_ratio,
        d=d.tolist(),
        y=y.tolist(),
        z=z.tolist(),
        b=b.tolist(),
        requested_file=req_file,
        current_time=10.0,
    )


def test_candidate_feature_free_space_capacity_units():
    f = 50
    resp = _make_state_resp(f, mem_ratio=0.12, req_file=1)  # 12% free of 5000 MiB = 600 MiB (0.6 GiB) >= 0.5 GiB
    vec_5000 = state_response_to_numpy(resp, f, use_candidate_features=True, cache_capacity=5000.0)
    assert len(vec_5000) == 10 + 1 + 5 * f
    assert vec_5000[8] == 1.0  # can_fit_free (600 MiB >= 500 MiB)
    assert vec_5000[9] == 0.0  # needed_space == 0

    resp_tight = _make_state_resp(f, mem_ratio=0.05, req_file=1)  # 5% of 5000 MiB = 250 MiB < 500 MiB
    vec_tight = state_response_to_numpy(resp_tight, f, use_candidate_features=True, cache_capacity=5000.0)
    assert vec_tight[8] == 0.0
    assert np.isclose(vec_tight[9], 0.25, atol=1e-5)  # Needs 250 MiB = 0.25 GiB


def test_expert_action_adalife_vs_part1():
    cfg_pro = AgentConfig(total_files=10, hidden_dim=32, adalife_pro=True, use_candidate_features=True)
    agent_pro = DDQLAgent(cfg_pro)

    # Full cache (can_fit_free=0), candidate not cached, negative density_diff but promising repeat candidate
    state = np.zeros(cfg_pro.state_dim, dtype=np.float32)
    state[0] = 0.25   # cand_pop >= 0.08
    state[3] = 0.70   # cand_density
    state[4] = 0.0    # not cached
    state[5] = 1.00   # min_cached_density (0.70 >= 0.5 * 1.00)
    state[7] = -0.30  # density_diff < 0
    state[8] = 0.0    # cache full
    assert agent_pro.compute_expert_action(state) == 1

    # Pure cold one-hit junk (cand_pop < 0.08, low density) -> rejected to protect Probation
    state[0] = 0.02
    state[3] = 0.20
    assert agent_pro.compute_expert_action(state) == 0

    # Part 1 strict mode rejects cand_density < 0.8 even with moderate popularity
    cfg_p1 = AgentConfig(total_files=10, hidden_dim=32, adalife_pro=False, use_candidate_features=True)
    agent_p1 = DDQLAgent(cfg_p1)
    state[0] = 0.25
    state[3] = 0.70
    assert agent_p1.compute_expert_action(state) == 0

