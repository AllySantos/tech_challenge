.PHONY: install

# Just simplifying project setup :)
install:
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Installing requirements..."
	@.venv/bin/pip3 install -q -r requirements.txt
	@echo "Done!"
