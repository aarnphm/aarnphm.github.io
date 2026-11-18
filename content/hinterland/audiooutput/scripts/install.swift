import Foundation

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let target = root.appendingPathComponent("dist/Audio Output.app", isDirectory: true)
let alias = URL(fileURLWithPath: "/Applications/Audio Output.app")

do {
    guard FileManager.default.isExecutableFile(atPath: target.appendingPathComponent("Contents/MacOS/AudioOutput").path) else {
        throw NSError(domain: "AudioOutputInstall", code: 1, userInfo: [NSLocalizedDescriptionKey: "Build Audio Output before installing its launcher."])
    }
    if FileManager.default.fileExists(atPath: alias.path) {
        let existing = try URL(resolvingAliasFileAt: alias, options: .withoutUI)
        guard existing.standardizedFileURL == target.standardizedFileURL else {
            throw NSError(domain: "AudioOutputInstall", code: 2, userInfo: [NSLocalizedDescriptionKey: "The Applications entry points elsewhere. It was left unchanged."])
        }
    } else {
        let bookmark = try target.bookmarkData(options: .suitableForBookmarkFile, includingResourceValuesForKeys: nil, relativeTo: nil)
        try URL.writeBookmarkData(bookmark, to: alias)
    }
    print("\(alias.path) → \(try URL(resolvingAliasFileAt: alias, options: .withoutUI).path)")
} catch {
    FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
    exit(1)
}
