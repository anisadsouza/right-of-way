#!/bin/bash
# Download a small subset of ESC-50 with curl, 8 files at a time.
# Safe to re-run: files already downloaded are skipped.
# Usage: bash fetch_esc50_subset.sh [clips_per_category] [dest_dir]
set -e
PER_CATEGORY="${1:-15}"
DEST="${2:-raw_datasets/ESC-50-master}"
BASE="https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
CATEGORIES="car_horn church_bells clock_alarm train helicopter crying_baby engine dog rain keyboard_typing siren"

mkdir -p "$DEST/audio" "$DEST/meta"
curl -fsS -o "$DEST/meta/esc50.csv" "$BASE/meta/esc50.csv"

for category in $CATEGORIES; do
  awk -F, -v c="$category" -v n="$PER_CATEGORY" '$4==c && count<n {print $1; count++}' "$DEST/meta/esc50.csv"
done > "$DEST/wanted.txt"

fetch_one() {
  file="$1"; dest="$2"; base="$3"
  [ -s "$dest/audio/$file" ] || curl -fsS --retry 3 -o "$dest/audio/$file" "$base/audio/$file"
}
export -f fetch_one

xargs -P 8 -I{} bash -c 'fetch_one "$@"' _ {} "$DEST" "$BASE" < "$DEST/wanted.txt"

echo "Wanted: $(wc -l < "$DEST/wanted.txt")  Downloaded: $(ls "$DEST/audio" | wc -l)"