import Darwin
import Foundation
import Security

public struct UploadReceipt: Sendable {
  public let activityID: String?
  public let duplicate: Bool

  public init(activityID: String?, duplicate: Bool) {
    self.activityID = activityID
    self.duplicate = duplicate
  }
}

public enum GarminConnectError: Error, LocalizedError {
  case authenticationRequired
  case invalidSession
  case credentialStorage(Int32)
  case invalidFile
  case rejected(Int)
  case malformedResponse
  case importFailed
  case uncertain
  case verificationFailed

  public var errorDescription: String? {
    switch self {
    case .authenticationRequired:
      return "Import a current Garmin session. Tether does not refresh shared credentials."
    case .invalidSession:
      return "The selected file does not contain a valid, unexpired Garmin access token."
    case .credentialStorage(let status):
      if status == errSecInteractionNotAllowed {
        return "Unlock Keychain or import your Garmin session again."
      }
      return "Keychain could not store or read the Garmin session (\(status))."
    case .invalidFile: return "Select a readable, nonempty FIT file smaller than 100 MB."
    case .rejected(let status): return "Garmin rejected the request (HTTP \(status))."
    case .malformedResponse: return "Garmin returned an unexpected response."
    case .importFailed:
      return "Garmin reported an import failure. The recording remains stored locally."
    case .uncertain:
      return
        "Upload outcome is uncertain. Check Garmin Connect before retrying; the recording remains stored locally."
    case .verificationFailed: return "The Garmin activity could not be verified."
    }
  }
}

private final class RejectRedirects: NSObject, URLSessionTaskDelegate {
  func urlSession(
    _ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    completionHandler(nil)
  }
}

public actor GarminConnectClient {
  private let credentials = CredentialStore()
  private var credentialTask: Task<GarminAccessCredential?, Error>?
  private let session: URLSession

  public init() {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.timeoutIntervalForRequest = 45
    configuration.timeoutIntervalForResource = 120
    configuration.httpCookieStorage = nil
    configuration.httpShouldSetCookies = false
    configuration.urlCache = nil
    session = URLSession(
      configuration: configuration, delegate: RejectRedirects(), delegateQueue: nil)
  }

  public func connectionStatus() async -> String {
    do {
      guard let credential = try await currentCredential() else { return "not connected" }
      guard credential.expiresAt.timeIntervalSinceNow > 60 else { return "session expired" }
      return "session saved"
    } catch { return "keychain unavailable" }
  }

  public func hasSession() async -> Bool {
    guard let credential = try? await currentCredential() else { return false }
    return credential.expiresAt.timeIntervalSinceNow > 60
  }

  public func refreshSession() async -> String {
    credentialTask = nil
    return await connectionStatus()
  }

  public func authorizeSession() async -> String {
    let store = credentials
    credentialTask = Task { try await store.read(interactive: true) }
    return await connectionStatus()
  }

  private func currentCredential() async throws -> GarminAccessCredential? {
    if let credentialTask { return try await credentialTask.value }
    let store = credentials
    let task = Task { try await store.read() }
    credentialTask = task
    return try await task.value
  }

  public func importSession(from url: URL) async throws {
    try Task.checkCancellation()
    let accessed = url.startAccessingSecurityScopedResource()
    defer { if accessed { url.stopAccessingSecurityScopedResource() } }
    guard url.isFileURL else { throw GarminConnectError.invalidSession }
    let values = try url.resourceValues(forKeys: [
      .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
    ])
    guard values.isRegularFile == true, values.isSymbolicLink != true,
      let size = values.fileSize, size > 0, size < 256 * 1024
    else {
      throw GarminConnectError.invalidSession
    }
    // Only the access token is imported; the original refresh owner keeps its token.
    let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW)
    guard descriptor >= 0 else { throw GarminConnectError.invalidSession }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? handle.close() }
    let data = try handle.read(upToCount: 256 * 1024) ?? Data()
    struct ImportedSession: Decodable {
      let diToken: String
      enum CodingKeys: String, CodingKey { case diToken = "di_token" }
    }
    struct Claims: Decodable { let exp: Double }
    guard let imported = try? JSONDecoder().decode(ImportedSession.self, from: data) else {
      throw GarminConnectError.invalidSession
    }
    let parts = imported.diToken.split(separator: ".", omittingEmptySubsequences: false)
    guard parts.count == 3 else { throw GarminConnectError.invalidSession }
    var encoded = String(parts[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(
      of: "_", with: "/")
    encoded += String(repeating: "=", count: (4 - encoded.count % 4) % 4)
    guard let payload = Data(base64Encoded: encoded),
      let claims = try? JSONDecoder().decode(Claims.self, from: payload), claims.exp.isFinite,
      claims.exp > Date().timeIntervalSince1970 + 60,
      !imported.diToken.contains(where: { $0.isWhitespace || $0.isNewline })
    else {
      throw GarminConnectError.invalidSession
    }
    let credential = GarminAccessCredential(
      token: imported.diToken, expiresAt: Date(timeIntervalSince1970: claims.exp))
    try await credentials.save(credential)
    credentialTask = Task { credential }
  }

  public func upload(file url: URL) async throws -> UploadReceipt {
    try Task.checkCancellation()
    guard url.isFileURL, url.pathExtension.lowercased() == "fit" else {
      throw GarminConnectError.invalidFile
    }
    let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard values.isRegularFile == true, let size = values.fileSize, size > 0, size < 100_000_000
    else {
      throw GarminConnectError.invalidFile
    }
    let bytes = try Data(contentsOf: url)
    try FITInspector.validateActivity(bytes)
    let boundary = "Tether-" + UUID().uuidString
    let filename = url.lastPathComponent.replacingOccurrences(of: "\"", with: "_")
      .replacingOccurrences(of: "\r", with: "_").replacingOccurrences(of: "\n", with: "_")
    var body = Data(
      "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        .utf8)
    body.append(bytes)
    body.append(Data("\r\n--\(boundary)--\r\n".utf8))
    var request = try await authenticatedRequest(path: "/upload-service/upload")
    request.httpMethod = "POST"
    request.setValue(
      "multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    request.httpBody = body
    let result: (Data, HTTPURLResponse)
    do { result = try await perform(request) } catch { throw GarminConnectError.uncertain }
    do {
      return try await withThrowingTaskGroup(of: UploadReceipt.self) { group in
        group.addTask { try await self.resolveImport(data: result.0, response: result.1) }
        group.addTask {
          try await Task.sleep(for: .seconds(240))
          throw GarminConnectError.uncertain
        }
        defer { group.cancelAll() }
        guard let receipt = try await group.next() else { throw GarminConnectError.uncertain }
        return receipt
      }
    } catch is CancellationError {
      throw GarminConnectError.uncertain
    }
  }

  public func reconcile(file: URL, filename: String) async throws -> String? {
    try await withThrowingTaskGroup(of: String?.self) { group in
      group.addTask { try await self.findExisting(file: file, filename: filename) }
      group.addTask {
        try await Task.sleep(for: .seconds(230))
        throw GarminConnectError.uncertain
      }
      defer { group.cancelAll() }
      return try await group.next() ?? nil
    }
  }

  private func findExisting(file: URL, filename: String) async throws -> String? {
    let id = try await GarminReconciliation.findMatch(file: file, filename: filename) { path in
      var request = try await self.authenticatedRequest(path: path)
      if path.hasPrefix("/download-service/") {
        request.setValue("*/*", forHTTPHeaderField: "Accept")
      }
      let (data, response) = try await self.perform(request)
      if response.statusCode == 401 || response.statusCode == 403 {
        throw GarminConnectError.authenticationRequired
      }
      guard response.statusCode == 200 else {
        throw GarminConnectError.rejected(response.statusCode)
      }
      return data
    }
    if let id { try await verify(activityID: id) }
    return id
  }

  public func verify(activityID: String) async throws {
    guard let number = UInt64(activityID), number > 0 else {
      throw GarminConnectError.verificationFailed
    }
    let request = try await authenticatedRequest(path: "/activity-service/activity/\(number)")
    let (data, response) = try await perform(request)
    try checkAuthorization(response)
    guard response.statusCode == 200 else { throw GarminConnectError.verificationFailed }
    struct Activity: Decodable { let activityId: UInt64 }
    guard let activity = try? JSONDecoder().decode(Activity.self, from: data),
      activity.activityId == number
    else {
      throw GarminConnectError.verificationFailed
    }
  }

  private func authenticatedRequest(path: String) async throws -> URLRequest {
    guard let url = URL(string: "https://connectapi.garmin.com" + path) else {
      throw GarminConnectError.malformedResponse
    }
    return try await authenticatedRequest(url: url)
  }

  private func authenticatedRequest(url: URL) async throws -> URLRequest {
    guard let credential = try await currentCredential(),
      credential.expiresAt.timeIntervalSinceNow > 60
    else {
      throw GarminConnectError.authenticationRequired
    }
    var request = URLRequest(url: url)
    request.setValue("Bearer \(credential.token)", forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.setValue("GCM-Android-5.23", forHTTPHeaderField: "User-Agent")
    request.setValue(
      "com.garmin.android.apps.connectmobile/5.23; ; Google/sdk_gphone64_arm64/google; Android/33; Dalvik/2.1.0",
      forHTTPHeaderField: "X-Garmin-User-Agent")
    request.setValue("10861", forHTTPHeaderField: "X-Garmin-Paired-App-Version")
    request.setValue("Android", forHTTPHeaderField: "X-Garmin-Client-Platform")
    request.setValue("10861", forHTTPHeaderField: "X-App-Ver")
    request.setValue("en", forHTTPHeaderField: "X-Lang")
    request.setValue("GC5", forHTTPHeaderField: "X-GCExperience")
    request.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")
    return request
  }

  private func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    try Task.checkCancellation()
    let (data, response) = try await session.data(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw GarminConnectError.malformedResponse
    }
    return (data, http)
  }

  private func checkAuthorization(_ response: HTTPURLResponse) throws {
    if response.statusCode == 401 || response.statusCode == 403 {
      throw GarminConnectError.authenticationRequired
    }
  }

  private func resolveImport(data: Data, response: HTTPURLResponse) async throws -> UploadReceipt {
    var data = data
    var response = response
    var pollURL: URL?
    for attempt in 0...60 {
      if pollURL != nil && (response.statusCode == 401 || response.statusCode == 403) {
        throw GarminConnectError.uncertain
      }
      try checkAuthorization(response)
      if response.statusCode == 409 { return duplicateReceipt(data: data) }
      if response.statusCode == 202 {
        guard attempt < 60 else { throw GarminConnectError.uncertain }
        if pollURL == nil {
          guard let location = response.value(forHTTPHeaderField: "Location"),
            let url = URL(string: location, relativeTo: response.url)?.absoluteURL,
            url.scheme == "https", url.host == "connectapi.garmin.com",
            url.port == nil || url.port == 443, url.user == nil, url.password == nil,
            url.path.hasPrefix("/upload-service/status/")
              || url.path.hasPrefix("/activity-service/activity/status/")
          else {
            throw GarminConnectError.uncertain
          }
          pollURL = url
        }
        let advertised =
          Double(response.value(forHTTPHeaderField: "Location-In-Milliseconds") ?? "") ?? 1000
        let delay = advertised.isFinite && advertised > 0 ? min(10000, max(1, advertised)) : 1000
        do {
          try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000))
          guard let pollURL else { throw GarminConnectError.uncertain }
          (data, response) = try await perform(authenticatedRequest(url: pollURL))
        } catch { throw GarminConnectError.uncertain }
        continue
      }
      guard response.statusCode == 200 || response.statusCode == 201 else {
        if pollURL != nil || response.statusCode >= 500 { throw GarminConnectError.uncertain }
        throw GarminConnectError.rejected(response.statusCode)
      }
      guard
        response.value(forHTTPHeaderField: "Content-Type")?.lowercased().contains("json") == true
      else {
        throw GarminConnectError.uncertain
      }
      struct Result: Decodable {
        struct Success: Decodable { let internalId: UInt64 }
        struct Failure: Decodable {
          let status: Int?
          let statusCode: Int?
          var isConflict: Bool { status == 409 || statusCode == 409 }
        }
        let successes: [Success]?
        let failures: [Failure]?
      }
      struct Envelope: Decodable {
        struct Nested: Decodable { let detailedImportResult: Result? }
        let detailedImportResult: Result?
        let data: Nested?
      }
      guard let envelope = try? JSONDecoder().decode(Envelope.self, from: data),
        let result = envelope.detailedImportResult ?? envelope.data?.detailedImportResult
      else {
        throw GarminConnectError.uncertain
      }
      if let failures = result.failures, !failures.isEmpty {
        guard result.successes?.isEmpty != false else { throw GarminConnectError.uncertain }
        if failures.count == 1, failures[0].isConflict, result.successes?.isEmpty != false {
          return UploadReceipt(activityID: nil, duplicate: true)
        }
        throw GarminConnectError.importFailed
      }
      guard let successes = result.successes, successes.count == 1,
        let success = successes.first, success.internalId > 0
      else { throw GarminConnectError.uncertain }
      return UploadReceipt(activityID: String(success.internalId), duplicate: false)
    }
    throw GarminConnectError.uncertain
  }

  private func duplicateReceipt(data: Data) -> UploadReceipt {
    struct ImportResult: Decodable {
      struct Success: Decodable {}
      struct Failure: Decodable {
        struct Message: Decodable { let code: Int }
        let internalId: UInt64
        let messages: [Message]
      }
      let successes: [Success]
      let failures: [Failure]
    }
    struct Envelope: Decodable {
      struct Nested: Decodable { let detailedImportResult: ImportResult? }
      let detailedImportResult: ImportResult?
      let data: Nested?
    }
    guard let envelope = try? JSONDecoder().decode(Envelope.self, from: data),
      let result = envelope.detailedImportResult ?? envelope.data?.detailedImportResult,
      result.successes.isEmpty, result.failures.count == 1,
      let failure = result.failures.first, failure.internalId > 0,
      !failure.messages.isEmpty, failure.messages.allSatisfy({ $0.code == 202 })
    else {
      return UploadReceipt(activityID: nil, duplicate: true)
    }
    return UploadReceipt(activityID: String(failure.internalId), duplicate: true)
  }
}
