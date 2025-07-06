.PHONY: lint

lint:
	@echo "Running isort..."
	isort .
	@echo "Running black..."
	black .
	@echo "Checking for trailing whitespace..."
	pre-commit run trailing-whitespace --all-files
	@echo "Fixing end of files..."
	pre-commit run end-of-file-fixer --all-files
