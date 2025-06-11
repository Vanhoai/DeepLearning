run:
	python main.py

test:
	pytest tests/ -s

install:
	pip install -r requirements.txt