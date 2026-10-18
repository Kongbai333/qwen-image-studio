# Contributing

1. Create a virtual environment and install `requirements-dev.txt`.
2. Run `pytest`.
3. Keep runtime files, tokens, proxies, and generated images out of commits.
4. Add tests for behavior changes that do not require a live Hugging Face
   Space.
5. Describe any change that affects the upstream `/generate` contract.

The legacy end-to-end scripts are intentionally not part of the default test
suite because they need a running service, real credentials, or generated
images.
