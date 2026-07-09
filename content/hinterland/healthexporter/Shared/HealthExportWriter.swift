import Foundation

struct HealthExportWriter {
  static let containerIdentifier = "iCloud.xyz.aarnphm.healthexporter"
  static let fileName = "apple-health-import.json"
  static let visiblePath = "iCloud Drive/HealthExporter/\(fileName)"

  private let fileManager: FileManager

  init(fileManager: FileManager = .default) {
    self.fileManager = fileManager
  }

  func write(_ document: HealthExportDocument) throws -> URL {
    let url = try exportURL()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    let data = try encoder.encode(document)
    try data.write(to: url, options: [.atomic])
    return url
  }

  func exportURL() throws -> URL {
    guard let container = fileManager.url(forUbiquityContainerIdentifier: Self.containerIdentifier)
    else {
      throw HealthExporterError.iCloudUnavailable(Self.containerIdentifier)
    }
    let documents = container.appendingPathComponent("Documents", isDirectory: true)
    try fileManager.createDirectory(at: documents, withIntermediateDirectories: true)
    return documents.appendingPathComponent(Self.fileName, isDirectory: false)
  }
}
