"""
gRPC Client communicating with the Go SMDP Edge Caching Simulation Server.
"""

from typing import Tuple, List, Optional
import numpy as np
import grpc

from pb import cache_env_pb2
from pb.cache_env_pb2_grpc import CacheEnvServiceStub


def state_response_to_numpy(
    resp: cache_env_pb2.StateResponse,
    total_files: int,
    use_candidate_features: bool = True,
) -> np.ndarray:
    """
    Transforms protobuf StateResponse into a normalized feature vector s(t).
    Base vector: [Mem(1), D(F), Y(F), Z(F)/1000, B(F), OneHot(RequestedFile)(F)] (1 + 5*F)
    Candidate features (10 dims): [cand_pop, cand_util, cand_size, cand_density, cand_is_cached,
                                   min_cached_density, avg_cached_density, density_diff,
                                   can_fit_free, needed_space]
    Resolves Agentic Gap #3: explicitly aligns candidate attributes with cache state.
    """
    mem = np.array([resp.mem], dtype=np.float32)
    d = np.array(resp.d, dtype=np.float32)
    y = np.array(resp.y, dtype=np.float32)
    z = np.array(resp.z, dtype=np.float32) / 1000.0  # Normalize size by 1000 MiB
    b = np.array(resp.b, dtype=np.float32)

    # One-hot encoded requested file index
    req_one_hot = np.zeros(total_files, dtype=np.float32)
    req_idx = resp.requested_file
    if 0 <= req_idx < total_files:
        req_one_hot[req_idx] = 1.0

    base_vec = np.concatenate([mem, d, y, z, b, req_one_hot])
    if not use_candidate_features:
        return base_vec

    # Explicit Candidate-Centric Features
    max_d = float(np.max(d)) if len(d) > 0 and np.max(d) > 0 else 1.0
    cand_pop = float(d[req_idx] / max_d) if (0 <= req_idx < len(d)) else 0.0
    cand_util = float(y[req_idx]) if (0 <= req_idx < len(y)) else 0.0
    cand_size = float(z[req_idx]) if (0 <= req_idx < len(z)) else 0.5
    cand_density = cand_util / max(0.01, cand_size)
    cand_is_cached = float(b[req_idx]) if (0 <= req_idx < len(b)) else 0.0

    cached_indices = np.where(b > 0.5)[0]
    if len(cached_indices) > 0:
        cached_densities = y[cached_indices] / np.maximum(0.01, z[cached_indices])
        min_cached_density = float(np.min(cached_densities))
        avg_cached_density = float(np.mean(cached_densities))
        density_diff = cand_density - min_cached_density
    else:
        min_cached_density = 0.0
        avg_cached_density = 0.0
        density_diff = cand_density

    free_mem = float(resp.mem)
    can_fit_free = 1.0 if free_mem >= cand_size else 0.0
    needed_space = max(0.0, cand_size - free_mem)

    cand_features = np.array([
        cand_pop,
        cand_util,
        cand_size,
        cand_density,
        cand_is_cached,
        min_cached_density,
        avg_cached_density,
        density_diff,
        can_fit_free,
        needed_space,
    ], dtype=np.float32)

    return np.concatenate([cand_features, base_vec])


class GrpcEnvClient:
    """Client for interacting with the Go SMDP Edge Caching gRPC environment."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        total_files: int = 50,
        use_candidate_features: bool = True,
    ):
        self.host = host
        self.port = port
        self.total_files = total_files
        self.use_candidate_features = use_candidate_features
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[CacheEnvServiceStub] = None
        self._connect()

    def _connect(self):
        address = f"{self.host}:{self.port}"
        # Max message length settings to ensure large state vectors transfer easily
        options = [
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ('grpc.max_send_message_length', 10 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(address, options=options)
        self.stub = CacheEnvServiceStub(self.channel)

    def reset(self, seed: int = 42, lambda_rate: float = 0.2, eta: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Resets environment state on the Go server with given parameters.
        Returns (state_vector, current_time).
        """
        request = cache_env_pb2.ResetRequest(
            seed=seed,
            eta=eta,
        )
        setattr(request, 'lambda', float(lambda_rate))
        resp = self.stub.Reset(request)
        state = state_response_to_numpy(resp, self.total_files, self.use_candidate_features)
        return state, resp.current_time

    def step(self, action: int) -> Tuple[np.ndarray, float, float, bool, float]:
        """
        Executes caching decision a(t) in {0, 1}.
        Returns (next_state, reward, tau, is_hit, utility_gained).
        """
        request = cache_env_pb2.StepRequest(action=int(action))
        resp = self.stub.Step(request)
        next_state = state_response_to_numpy(resp.next_state, self.total_files, self.use_candidate_features)
        return (
            next_state,
            resp.reward,
            resp.tau,
            resp.is_hit,
            resp.utility_gained,
        )

    def batch_step(self, actions: List[int]) -> List[Tuple[np.ndarray, float, float, bool, float]]:
        """Executes a sequence of actions in batch on the server."""
        request = cache_env_pb2.BatchStepRequest(actions=[int(a) for a in actions])
        resp = self.stub.BatchStep(request)
        results = []
        for step_resp in resp.step_responses:
            next_state = state_response_to_numpy(step_resp.next_state, self.total_files, self.use_candidate_features)
            results.append((
                next_state,
                step_resp.reward,
                step_resp.tau,
                step_resp.is_hit,
                step_resp.utility_gained,
            ))
        return results

    def close(self):
        """Closes gRPC communication channel."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
