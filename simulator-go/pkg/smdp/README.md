# SMDP processes

This package generates the stochastic parts of an experiment. Poisson sampling produces inter-arrival times, and Zipf sampling produces requested file IDs according to configured popularity.

The generator is seeded by callers. Reusing the same seed, configuration, and request count produces the same request sequence, which is required for fair policy comparisons.
