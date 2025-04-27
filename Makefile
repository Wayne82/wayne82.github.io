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

# Build the Jekyll site
build:
	@echo "Building Jekyll site..."
	bundle exec jekyll build

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

GALLERIES_COLLECTION=_galleries
add-new-gallery:
ifndef name
	$(error ❌ Please provide a gallery name: 'make add-new-gallery name=your-gallery-name')
endif
	@echo "Creating new gallery: $(name)"
	@mkdir -p $(GALLERY_FOLDER)/$(name)
	@mkdir -p $(DATA_FOLDER)
	@mkdir -p $(GALLERIES_COLLECTION)
	@echo "---" > $(GALLERIES_COLLECTION)/$(name).md
	@echo "layout: gallery" >> $(GALLERIES_COLLECTION)/$(name).md
	@echo "title: \"$(name)\"" >> $(GALLERIES_COLLECTION)/$(name).md
	@echo "data: \"$(name)\"" >> $(GALLERIES_COLLECTION)/$(name).md
	@echo "---" >> $(GALLERIES_COLLECTION)/$(name).md
	@echo "✅ New gallery '$(name)' created! Please add images into $(GALLERY_FOLDER)/$(name) and update the metadata using the update-gallery command."