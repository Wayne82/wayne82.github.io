#!/usr/bin/env bash

# Usage: run at the root of the site.
# gallery_image_gen.sh <gallery_folder> <output_yaml>
# E.g. ./scripts/gallery_image_gen.sh assets/galleries/sketch/ _data/galleries/sketch.yml

set -e

GALLERY_DIR="$1"
OUTPUT_YAML="$2"

THUMB_SUFFIX="-thumb"
THUMB_WIDTH=392

# List of supported image extensions (lowercase)
IMAGE_EXTENSIONS=("jpg" "jpeg" "png" "webp")

if [ ! -d "$GALLERY_DIR" ]; then
  echo "Error: Gallery folder not found: $GALLERY_DIR"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_YAML")"

# Create YAML if not exist
if [ ! -f "$OUTPUT_YAML" ]; then
  echo "Creating new YAML: $OUTPUT_YAML"
  echo "path: \"/$GALLERY_DIR\"" > "$OUTPUT_YAML"
  echo "cover_image: 0" >> "$OUTPUT_YAML"
  echo "images:" >> "$OUTPUT_YAML"
fi

# Read existing image names
EXISTING_IMAGES=$(grep 'name:' "$OUTPUT_YAML" | sed -E 's/.*name: "(.*)"/\1/')

NEW_IMAGES_ADDED=0

# Loop over all supported images
for EXT in "${IMAGE_EXTENSIONS[@]}"; do
  for IMG_PATH in "$GALLERY_DIR"/*."$EXT"; do
    [ -e "$IMG_PATH" ] || continue

    FILENAME=$(basename "$IMG_PATH")

    # Skip thumbnails
    if [[ "$FILENAME" == *"$THUMB_SUFFIX."* ]]; then
      continue
    fi

    # Skip if already exists
    if echo "$EXISTING_IMAGES" | grep -qF "$FILENAME"; then
      echo "Skipping existing: $FILENAME"
      continue
    fi

    echo "Processing new image: $FILENAME"

    BASENAME="${FILENAME%.*}"
    EXTENSION="${FILENAME##*.}"
    THUMB_FILENAME="${BASENAME}${THUMB_SUFFIX}.${EXTENSION}"
    THUMB_PATH="$GALLERY_DIR/$THUMB_FILENAME"

    # Create thumbnail if missing
    if [ ! -f "$THUMB_PATH" ]; then
      echo "Creating thumbnail: $THUMB_FILENAME"
      magick "$IMG_PATH" -resize "${THUMB_WIDTH}x" "$THUMB_PATH"
    fi

    # Get width and height
    DIMENSIONS=$(magick identify -format "%w %h" "$IMG_PATH")
    WIDTH=$(echo $DIMENSIONS | cut -d' ' -f1)
    HEIGHT=$(echo $DIMENSIONS | cut -d' ' -f2)

    # Generate alt text
    # Generate alt text: Title Case
    ALT_TEXT=$(basename "$FILENAME" ".$EXTENSION" | tr '-' ' ')
    ALT_TEXT=$(echo "$ALT_TEXT" | awk '{ for (i=1;i<=NF;i++) { $i=toupper(substr($i,1,1)) substr($i,2) } print }')

    # Append directly to YAML
    cat >> "$OUTPUT_YAML" <<EOF
  - name: "$FILENAME"
    thumb: "$THUMB_FILENAME"
    width: $WIDTH
    height: $HEIGHT
    alt: "$ALT_TEXT"
EOF

    NEW_IMAGES_ADDED=$((NEW_IMAGES_ADDED + 1))
  done
done

# Final report
if [ $NEW_IMAGES_ADDED -gt 0 ]; then
  echo "✅ Added $NEW_IMAGES_ADDED new images!"
else
  echo "ℹ️  No new images to add."
fi
