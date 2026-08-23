# gRPC server

This package adapts the native cache engine to the protobuf environment service. It handles remote reset and step-style interactions, converts simulator state into protobuf messages, and keeps transport concerns outside the core model.

The command entry point is `cmd/server`; this package is not intended to be run directly.
