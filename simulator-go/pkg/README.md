# Simulator packages

The packages here contain the simulator's reusable domain logic. Commands under `cmd` should compose these packages rather than duplicate their behavior.

The normal data path is: configuration -> generated files and request process -> cache policy or core engine -> reward/state -> optional gRPC transport.
