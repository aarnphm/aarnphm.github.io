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
codesign --force --sign "${AUDIOOUTPUT_SIGN_IDENTITY:--}" --identifier xyz.aarnphm.audiooutput "$app"
codesign --verify --strict "$app"
print -r -- "$app"
