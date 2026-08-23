# Environment server

`main.go` starts the native cache environment and serves it through gRPC. It is the bridge for clients that need to step the simulator remotely, including the Python agent.

The server uses the default configuration and generated files. The protobuf service contract is defined in the repository `proto` directory; regenerate bindings before changing that contract.
