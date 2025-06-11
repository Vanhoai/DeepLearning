run:
	python main.py

test_neural_network_v1:
	python -m unittest tests/test_neural_network_v1.py

test_neural_network_v2:
	python -m unittest tests/test_neural_network_v2.py

test_neural_network_v3:
	python -m unittest tests/test_neural_network_v3.py

test_neural_network:
	python -m unittest discover -s tests -p "test_neural_network_*.py"

install:
	pip install -r requirements.txt