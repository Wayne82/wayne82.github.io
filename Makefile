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

# Variables
GALLERY_FOLDER=assets/galleries
DATA_FOLDER=_data/galleries
SCRIPT=scripts/gallery_image_gen.sh

update-gallery:
	@echo "Updating gallery metadata and generate thumbnails if missing ..."
	@for dir in $(GALLERY_FOLDER)/*/ ; do \
	  gallery_name=$$(basename $$dir) ; \
	  yaml_file=$(DATA_FOLDER)/$$gallery_name.yml ; \
	  echo "Running: $(SCRIPT) $$dir $$yaml_file" ; \
	  bash $(SCRIPT) "$$dir" "$$yaml_file" ; \
	done
	@echo "✅ Gallery metadata update and thumbnails generate complete!"
