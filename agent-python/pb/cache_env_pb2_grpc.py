# Generated gRPC stubs for cache_env.proto
import grpc
from pb import cache_env_pb2 as pb__cache__env__pb2

class CacheEnvServiceStub:
    """Service stub for CacheEnvService in Go SMDP edge-caching simulator."""

    def __init__(self, channel: grpc.Channel):
        self.Reset = channel.unary_unary(
            '/cache_env.CacheEnvService/Reset',
            request_serializer=pb__cache__env__pb2.ResetRequest.SerializeToString,
            response_deserializer=pb__cache__env__pb2.StateResponse.FromString,
        )
        self.Step = channel.unary_unary(
            '/cache_env.CacheEnvService/Step',
            request_serializer=pb__cache__env__pb2.StepRequest.SerializeToString,
            response_deserializer=pb__cache__env__pb2.StepResponse.FromString,
        )
        self.BatchStep = channel.unary_unary(
            '/cache_env.CacheEnvService/BatchStep',
            request_serializer=pb__cache__env__pb2.BatchStepRequest.SerializeToString,
            response_deserializer=pb__cache__env__pb2.BatchStepResponse.FromString,
        )

class CacheEnvServiceServicer:
    """Base servicer interface for CacheEnvService."""

    def Reset(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Step(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BatchStep(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_CacheEnvServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Reset': grpc.unary_unary_rpc_method_handler(
            servicer.Reset,
            request_deserializer=pb__cache__env__pb2.ResetRequest.FromString,
            response_serializer=pb__cache__env__pb2.StateResponse.SerializeToString,
        ),
        'Step': grpc.unary_unary_rpc_method_handler(
            servicer.Step,
            request_deserializer=pb__cache__env__pb2.StepRequest.FromString,
            response_serializer=pb__cache__env__pb2.StepResponse.SerializeToString,
        ),
        'BatchStep': grpc.unary_unary_rpc_method_handler(
            servicer.BatchStep,
            request_deserializer=pb__cache__env__pb2.BatchStepRequest.FromString,
            response_serializer=pb__cache__env__pb2.BatchStepResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'cache_env.CacheEnvService', rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))

