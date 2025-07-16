sync:
	uv sync --all-packages

benchmark:
	uv run benchmark

dl:
	uv run dl

training:
	uv run training

inference:
	uv run inference