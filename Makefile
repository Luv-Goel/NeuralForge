.PHONY: install test lint clean build

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=neuralforge

test-quick:
	pytest tests/ -x -q

lint:
	flake8 neuralforge/ tests/
	black --check neuralforge/ tests/

format:
	black neuralforge/ tests/
	isort neuralforge/ tests/

clean:
	rm -rf build/ dist/ *.egg-info __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

build: clean
	python -m build

docs:
	python -c "import neuralforge; print(f'NeuralForge v{neuralforge.__version__}')"
