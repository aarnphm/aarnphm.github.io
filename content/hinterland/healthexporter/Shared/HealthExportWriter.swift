import Foundation

struct HealthExportWriter {
  static let containerIdentifier = "iCloud.xyz.aarnphm.healthexporter"
  static let fileName = "apple-health-import.json"
  static let visiblePath = "iCloud Drive/HealthExporter/\(fileName)"

  private let fileManager: FileManager
  private let containerURL: URL?

  init(fileManager: FileManager = .default, containerURL: URL? = nil) {
    self.fileManager = fileManager
    self.containerURL = containerURL
  }

  func write(_ document: HealthExportDocument) throws -> URL {
    let url = try exportURL()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    let data = try encoder.encode(document)
    try data.write(to: url, options: [.atomic])
    return url
  }

  func read() throws -> HealthExportResult? {
    let url = try exportURL()
    guard fileManager.fileExists(atPath: url.path) else { return nil }
    let data = try Data(contentsOf: url, options: [.mappedIfSafe])
    let document = try JSONDecoder().decode(HealthExportDocument.self, from: data)
    return HealthExportResult(document: document, url: url)
  }

  func exportURL() throws -> URL {
    guard let container = containerURL
      ?? fileManager.url(forUbiquityContainerIdentifier: Self.containerIdentifier)
    else {
      throw HealthExporterError.iCloudUnavailable(Self.containerIdentifier)
    }
    let documents = container.appendingPathComponent("Documents", isDirectory: true)
    try fileManager.createDirectory(at: documents, withIntermediateDirectories: true)
    return documents.appendingPathComponent(Self.fileName, isDirectory: false)
  }
}
