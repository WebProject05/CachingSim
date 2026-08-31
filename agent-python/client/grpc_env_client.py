"""
gRPC Client communicating with the Go SMDP Edge Caching Simulation Server.
"""

from typing import Tuple, List, Optional
import numpy as np
import grpc

from pb import cache_env_pb2
from pb.cache_env_pb2_grpc import CacheEnvServiceStub


def state_response_to_numpy(resp: cache_env_pb2.StateResponse, total_files: int) -> np.ndarray:
    """
    Transforms protobuf StateResponse into a normalized feature vector s(t):
    [Mem(1), D(F), Y(F), Z(F)/1000, B(F), OneHot(RequestedFile)(F)]
    Total length = 1 + 5*F (251 for F=50).
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

    return np.concatenate([mem, d, y, z, b, req_one_hot])


class GrpcEnvClient:
    """Client for interacting with the Go SMDP Edge Caching gRPC environment."""

    def __init__(self, host: str = "localhost", port: int = 50051, total_files: int = 50):
        self.host = host
        self.port = port
        self.total_files = total_files
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
        state = state_response_to_numpy(resp, self.total_files)
        return state, resp.current_time

    def step(self, action: int) -> Tuple[np.ndarray, float, float, bool, float]:
        """
        Executes caching decision a(t) in {0, 1}.
        Returns (next_state, reward, tau, is_hit, utility_gained).
        """
        request = cache_env_pb2.StepRequest(action=int(action))
        resp = self.stub.Step(request)
        next_state = state_response_to_numpy(resp.next_state, self.total_files)
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
            next_state = state_response_to_numpy(step_resp.next_state, self.total_files)
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
