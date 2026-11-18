// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Tether",
  platforms: [.macOS(.v14)],
  products: [
    .executable(name: "Tether", targets: ["Tether"]),
    .executable(name: "tether-probe", targets: ["TetherProbe"]),
    .executable(name: "tether-mtp", targets: ["TetherMTP"]),
  ],
  targets: [
    .target(name: "TetherCore"),
    .testTarget(name: "TetherCoreTests", dependencies: ["TetherCore"]),
    .executableTarget(name: "TetherMTP"),
    .executableTarget(name: "Tether", dependencies: ["TetherCore"]),
    .executableTarget(name: "TetherProbe", dependencies: ["TetherCore"]),
  ],
  swiftLanguageModes: [.v5]
)
