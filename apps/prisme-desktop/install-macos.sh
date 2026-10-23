#!/bin/sh
# Builds Prisme and installs it as /Applications/Prisme.app (Spotlight, Launchpad, Dock).
# Run it again to update; `--pull` fetches the latest commits first.
#   apps/prisme-desktop/install-macos.sh [--pull]
# PRISME_APP_DIR overrides the destination folder (default /Applications).
set -eu

root="$(cd "$(dirname "$0")/../.." && pwd)"
dest_dir="${PRISME_APP_DIR:-/Applications}"
app="$dest_dir/Prisme.app"

cd "$root"
if [ "${1:-}" = "--pull" ]; then
    git pull --ff-only
fi

cargo build --release -p prisme-desktop

version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
commit="$(git rev-parse --short HEAD)"
dirty="$(git diff --quiet HEAD -- . 2>/dev/null || echo '+')"

# Assemble next to the destination, then swap it in.
tmp="$(mktemp -d "$dest_dir/.Prisme.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
bundle="$tmp/Prisme.app"
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
cp target/release/prisme "$bundle/Contents/MacOS/prisme"

iconset="$tmp/Prisme.iconset"
mkdir "$iconset"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" apps/prisme-desktop/icon.png --out "$iconset/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" apps/prisme-desktop/icon.png --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$bundle/Contents/Resources/Prisme.icns"

cat > "$bundle/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Prisme</string>
    <key>CFBundleDisplayName</key><string>Prisme</string>
    <key>CFBundleIdentifier</key><string>com.sigma-studios.prisme</string>
    <key>CFBundleExecutable</key><string>prisme</string>
    <key>CFBundleIconFile</key><string>Prisme</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleVersion</key><string>$version ($commit$dirty)</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

codesign --force --sign - "$bundle"

rm -rf "$app"
mv "$bundle" "$app"
# Let Launch Services and Spotlight pick up the new icon and version right away.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$app"
mdimport "$app" 2>/dev/null || true

echo "Installed $app ($version, $commit$dirty)"
