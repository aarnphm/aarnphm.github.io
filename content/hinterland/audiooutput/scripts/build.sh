#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
swift build -c release
app="$PWD/dist/Audio Output.app"
mkdir -p "$app/Contents/MacOS"
cp .build/release/AudioOutput "$app/Contents/MacOS/AudioOutput"
mkdir -p "$app/Contents/Resources"
cp -R Extension "$app/Contents/Resources/"
cp Info.plist "$app/Contents/Info.plist"
identity="${AUDIOOUTPUT_SIGN_IDENTITY:-}"
if [[ -z "$identity" ]]; then
    identities=("${(@f)$(security find-identity -v -p codesigning | sed -nE 's/^[[:space:]]*[0-9]+\) ([A-F0-9]+) ".*/\1/p')}")
    if (( ${#identities} == 1 )) && [[ -n "${identities[1]}" ]]; then
        identity="${identities[1]}"
    else
        print -u2 'Set AUDIOOUTPUT_SIGN_IDENTITY to a signing identity. Use - explicitly for an ad-hoc build that loses permissions after changes.'
        exit 1
    fi
fi
codesign --force --sign "$identity" --identifier xyz.aarnphm.audiooutput "$app"
codesign --verify --strict "$app"
print -r -- "$app"
