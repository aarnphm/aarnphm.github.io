#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

signing_identity="${TETHER_SIGN_IDENTITY:-}"
if [[ -z "$signing_identity" ]]; then
    available_identities="$(/usr/bin/security find-identity -v -p codesigning)"
    identity_hashes="$(printf '%s\n' "$available_identities" | /usr/bin/sed -nE 's/^[[:space:]]*[0-9]+\) ([A-Fa-f0-9]{40}) ".*/\1/p')"
    identities=()
    while IFS= read -r identity_hash; do
        if [[ -n "$identity_hash" ]]; then
            identities+=("$identity_hash")
        fi
    done <<< "$identity_hashes"
    if [[ ${#identities[@]} -ne 1 ]]; then
        printf 'Set TETHER_SIGN_IDENTITY to an installed code-signing identity; found %s valid identities.\n' "${#identities[@]}" >&2
        exit 1
    fi
    signing_identity="${identities[0]}"
fi
if [[ "$signing_identity" == "-" ]]; then
    printf 'Tether requires a certificate signing identity so Keychain access survives rebuilds.\n' >&2
    exit 1
fi

swift build --configuration release --product Tether
swift build --configuration release --product tether-mtp
binary_directory="$(swift build --configuration release --show-bin-path)"

mkdir -p "$project_root/dist"
staging_directory="$(mktemp -d "$project_root/dist/.tether-stage.XXXXXX")"
previous_directory=""
app_destination="$project_root/dist/Tether.app"

cleanup() {
    local result=$?
    if [[ $result -ne 0 && -n "$previous_directory" && ! -e "$app_destination" ]]; then
        if ! mv "$previous_directory/Tether.app" "$app_destination"; then
            printf 'Previous app remains at %s/Tether.app\n' "$previous_directory" >&2
        fi
    fi
    rm -rf "$staging_directory"
    exit "$result"
}
trap cleanup EXIT

app_bundle="$staging_directory/Tether.app"
mkdir -p "$app_bundle/Contents/MacOS" "$app_bundle/Contents/Resources"
cp "$project_root/Resources/AppIcon.icns" "$app_bundle/Contents/Resources/AppIcon.icns"
cp "$project_root/Info.plist" "$app_bundle/Contents/Info.plist"
install -m 755 "$binary_directory/Tether" "$app_bundle/Contents/MacOS/Tether"
install -m 755 "$binary_directory/tether-mtp" "$app_bundle/Contents/MacOS/tether-mtp"

/usr/bin/plutil -lint "$app_bundle/Contents/Info.plist"
/usr/bin/codesign --force --sign "$signing_identity" --identifier app.aarnphm.tether.mtp "$app_bundle/Contents/MacOS/tether-mtp"
/usr/bin/codesign --force --sign "$signing_identity" --identifier app.aarnphm.tether "$app_bundle"
/usr/bin/codesign --verify --deep --strict "$app_bundle"

if [[ -e "$app_destination" ]]; then
    previous_directory="$(mktemp -d "$project_root/dist/previous-$(date +%Y%m%d-%H%M%S).XXXXXX")"
    mv "$app_destination" "$previous_directory/Tether.app"
fi
mv "$app_bundle" "$app_destination"
printf 'Built %s\n' "$app_destination"
if [[ -n "$previous_directory" ]]; then
    printf 'Previous app preserved at %s/Tether.app\n' "$previous_directory"
fi
