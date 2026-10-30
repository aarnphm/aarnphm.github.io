import Foundation

struct SiteRule: Codable, Identifiable, Equatable {
    var id: String
    var name: String
    var outputUID: String
    var outputName: String
    var volume: Double? = nil
    var gain: Double { min(3, max(0, volume ?? 1)) }
}

struct SiteConfiguration: Codable {
    var enabled: Bool
    var sites: [SiteRule]
}

struct SiteStatus: Codable {
    var host: String
    var state: String
    var detail: String
    var outputUID: String
    var updatedAt: Date
}

enum SiteRouting {
    static let directory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/AudioOutput", isDirectory: true)
    static let configurationURL = directory.appendingPathComponent("sites.json")

    static func load() -> SiteConfiguration {
        guard let data = try? Data(contentsOf: configurationURL),
              let configuration = try? JSONDecoder().decode(SiteConfiguration.self, from: data) else {
            return SiteConfiguration(enabled: true, sites: [SiteRule(
                id: "music.youtube.com", name: "YouTube Music", outputUID: "", outputName: "Normal output")])
        }
        return configuration
    }

    static func save(_ configuration: SiteConfiguration) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(configuration).write(to: configurationURL, options: .atomic)
    }

    static func snapshot() -> SiteConfiguration {
        var configuration = load()
        let attributes = try? FileManager.default.attributesOfItem(atPath: configurationURL.path)
        let modified = attributes?[.modificationDate] as? Date
        if modified == nil || Date().timeIntervalSince(modified ?? .distantPast) > 10 { configuration.enabled = false }
        return configuration
    }

    static func status(_ host: String) -> SiteStatus? {
        guard validHost(host),
              let data = try? Data(contentsOf: directory.appendingPathComponent("status-\(host).json")),
              let value = try? JSONDecoder().decode(SiteStatus.self, from: data),
              Date().timeIntervalSince(value.updatedAt) < 10 else { return nil }
        return value
    }

    static func report(_ status: SiteStatus) throws {
        guard validHost(status.host) else { throw AudioFailure(message: "Invalid site hostname") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(status).write(to: directory.appendingPathComponent("status-\(status.host).json"), options: .atomic)
    }

    static func validHost(_ host: String) -> Bool {
        !host.isEmpty && host.count <= 253 && host.range(of: "^[a-z0-9]+([.-][a-z0-9]+)*$", options: .regularExpression) != nil
    }
}
