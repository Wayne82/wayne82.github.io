# Makefile for Jekyll commands

.PHONY: serve serve-drafts

# Default command (run when you just type 'make')
default: serve

# Run Jekyll server with drafts enabled by default
serve:
	@echo "Starting Jekyll server with drafts..."
	bundle exec jekyll serve --drafts

# Run Jekyll server with standard configuration
serve-prod:
	@echo "Starting Jekyll server..."
	bundle exec jekyll serve

# Clean the Jekyll build artifacts
clean:
	@echo "Cleaning Jekyll build..."
	bundle exec jekyll clean