// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AudioOutput",
    platforms: [.macOS("14.2")],
    targets: [
        .target(name: "AudioRender", publicHeadersPath: "include"),
        .executableTarget(name: "AudioOutput", dependencies: ["AudioRender"]),
    ],
    swiftLanguageModes: [.v5]
)
