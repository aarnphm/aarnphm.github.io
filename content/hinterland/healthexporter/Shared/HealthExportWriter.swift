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

  func write(_ document: HealthExportDocument, routes: [WorkoutRouteValue] = []) throws -> URL {
    let url = try exportURL()
    try writeRoutes(routes, documentsURL: url.deletingLastPathComponent())
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

  private func writeRoutes(_ routes: [WorkoutRouteValue], documentsURL: URL) throws {
    guard !routes.isEmpty else { return }
    let directory = documentsURL.appendingPathComponent("GPX", isDirectory: true)
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    for route in routes where !route.points.isEmpty {
      let url = documentsURL.appendingPathComponent(route.relativePath, isDirectory: false)
      try gpxData(route).write(to: url, options: [.atomic])
    }
  }

  private func gpxData(_ route: WorkoutRouteValue) -> Data {
    let lines = [
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
      "<gpx creator=\"HealthExporter\" version=\"1.1\" xmlns=\"http://www.topografix.com/GPX/1/1\" xmlns:gpxtpx=\"http://www.garmin.com/xmlschemas/TrackPointExtension/v1\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd http://www.garmin.com/xmlschemas/TrackPointExtension/v1 http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd\">",
      " <metadata>",
      "  <time>\(HealthExporterFormat.utcTimestampString(route.start))</time>",
      " </metadata>",
      " <trk>",
      "  <name>\(route.activity)</name>",
      "  <type>\(route.activity)</type>",
      "  <trkseg>",
      route.points.map(gpxTrackPoint).joined(separator: "\n"),
      "  </trkseg>",
      " </trk>",
      "</gpx>",
      "",
    ]
    return Data(lines.joined(separator: "\n").utf8)
  }

  private func gpxTrackPoint(_ point: WorkoutRoutePointValue) -> String {
    var lines = [
      "   <trkpt lat=\"\(decimal(point.latitude, places: 7))\" lon=\"\(decimal(point.longitude, places: 7))\">",
    ]
    if let altitudeM = point.altitudeM {
      lines.append("    <ele>\(decimal(altitudeM, places: 1))</ele>")
    }
    lines.append("    <time>\(HealthExporterFormat.utcTimestampString(point.date))</time>")
    if point.heartRateBpm != nil || point.cadenceSpm != nil || point.powerW != nil {
      lines.append("    <extensions>")
      if let powerW = point.powerW {
        lines.append("     <power>\(Int(powerW.rounded()))</power>")
      }
      if point.heartRateBpm != nil || point.cadenceSpm != nil {
        lines.append("     <gpxtpx:TrackPointExtension>")
        if let heartRateBpm = point.heartRateBpm {
          lines.append("      <gpxtpx:hr>\(heartRateBpm)</gpxtpx:hr>")
        }
        if let cadenceSpm = point.cadenceSpm {
          lines.append("      <gpxtpx:cad>\(Int((cadenceSpm / 2).rounded()))</gpxtpx:cad>")
        }
        lines.append("     </gpxtpx:TrackPointExtension>")
      }
      lines.append("    </extensions>")
    }
    lines.append("   </trkpt>")
    return lines.joined(separator: "\n")
  }

  private func decimal(_ value: Double, places: Int) -> String {
    String(format: "%.*f", locale: Locale(identifier: "en_US_POSIX"), places, value)
  }
}
