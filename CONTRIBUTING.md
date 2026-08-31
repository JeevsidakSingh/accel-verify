# Contributing

Accel-Verify is early-stage. Small, evidence-backed contributions are preferred over broad abstractions.

## Good contributions

- A minimized public correctness reproducer with a source issue link
- A failing test for incorrect output, shape, dtype, or input gradients
- Improvements to reports or environment capture
- Fixes that preserve deterministic reference-versus-candidate execution

Do not submit proprietary workloads, production data, credentials, model weights, or code you are not authorized to publish.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Pull requests should explain:

1. The semantic contract being tested.
2. The reference and candidate implementations.
3. The relevant framework, compiler, hardware, dtype, and shape.
4. Why the configured tolerance is meaningful.
5. Whether the behavior is current, historical, or intentionally seeded.

Public issue repros must link to their original source and avoid including unrelated project code.
