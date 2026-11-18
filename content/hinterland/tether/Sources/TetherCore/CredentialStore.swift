import Foundation
@preconcurrency import LocalAuthentication
import Security

struct GarminAccessCredential: Codable, Sendable {
  let token: String
  let expiresAt: Date
}

struct CredentialStore {
  private let service = "app.aarnphm.tether.garmin"
  private let account = "imported-access-token"

  func read(interactive: Bool = false) async throws -> GarminAccessCredential? {
    try await boundedOperation(interactive: interactive) { context in try read(context: context) }
  }

  private func read(context: LAContext) throws -> GarminAccessCredential? {
    var query = identity
    query[kSecUseAuthenticationContext] = context
    query[kSecReturnData] = true
    query[kSecMatchLimit] = kSecMatchLimitOne
    var result: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &result)
    if status == errSecItemNotFound { return nil }
    guard status == errSecSuccess, let data = result as? Data else {
      throw GarminConnectError.credentialStorage(status)
    }
    guard let credential = try? JSONDecoder().decode(GarminAccessCredential.self, from: data) else {
      throw GarminConnectError.invalidSession
    }
    return credential
  }

  func save(_ credential: GarminAccessCredential) async throws {
    _ = try await boundedOperation(interactive: true) { context in
      try save(credential, context: context)
      return credential
    }
  }

  private func save(_ credential: GarminAccessCredential, context: LAContext) throws {
    let data = try JSONEncoder().encode(credential)
    let update = [kSecValueData: data] as CFDictionary
    var query = identity
    query[kSecUseAuthenticationContext] = context
    let status = SecItemUpdate(query as CFDictionary, update)
    if status == errSecItemNotFound {
      var item = query
      item[kSecValueData] = data
      item[kSecAttrAccessible] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
      let added = SecItemAdd(item as CFDictionary, nil)
      guard added == errSecSuccess else { throw GarminConnectError.credentialStorage(added) }
    } else if status != errSecSuccess {
      throw GarminConnectError.credentialStorage(status)
    }
  }

  private func boundedOperation(
    interactive: Bool,
    operation: @escaping @Sendable (LAContext) throws -> GarminAccessCredential?
  ) async throws -> GarminAccessCredential? {
    let context = LAContext()
    context.interactionNotAllowed = !interactive
    return try await withCheckedThrowingContinuation { continuation in
      let completion = KeychainCompletion(continuation)
      // Security.framework is synchronous and can outlive LAContext cancellation on legacy keychains.
      DispatchQueue.global(qos: .utility).async {
        completion.finish(Result { try operation(context) })
      }
      DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + (interactive ? 120 : 2)) {
        if completion.finish(
          .failure(GarminConnectError.credentialStorage(errSecInteractionNotAllowed)))
        {
          context.invalidate()
        }
      }
    }
  }

  private var identity: [CFString: Any] {
    [
      kSecClass: kSecClassGenericPassword, kSecAttrService: service,
      kSecAttrAccount: account, kSecAttrSynchronizable: false,
    ]
  }
}

private final class KeychainCompletion: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<GarminAccessCredential?, Error>?

  init(_ continuation: CheckedContinuation<GarminAccessCredential?, Error>) {
    self.continuation = continuation
  }

  @discardableResult
  func finish(_ result: Result<GarminAccessCredential?, Error>) -> Bool {
    lock.lock()
    let pending = continuation
    continuation = nil
    lock.unlock()
    pending?.resume(with: result)
    return pending != nil
  }
}
