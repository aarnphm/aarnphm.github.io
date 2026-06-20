import Foundation
import WatchConnectivity

final class PhoneWatchSession: NSObject, WCSessionDelegate {
  static let shared = PhoneWatchSession()

  private var exportHandler: (() -> Void)?

  private override init() {
    super.init()
  }

  func start(exportHandler: @escaping () -> Void) {
    guard WCSession.isSupported() else { return }
    self.exportHandler = exportHandler
    let session = WCSession.default
    session.delegate = self
    session.activate()
  }

  func publish(_ result: HealthExportResult) {
    guard WCSession.isSupported(), WCSession.default.activationState == .activated else { return }
    let context: [String: Any] = [
      "generatedAt": result.document.generatedAt,
      "dayCount": result.document.days.count,
      "swimCount": result.document.swims.count,
      "path": result.url.path,
    ]
    try? WCSession.default.updateApplicationContext(context)
  }

  private func handle(_ message: [String: Any], replyHandler: (([String: Any]) -> Void)? = nil) {
    guard message["command"] as? String == "export" else {
      replyHandler?(["accepted": false])
      return
    }
    exportHandler?()
    replyHandler?(["accepted": true])
  }

  func session(
    _ session: WCSession,
    activationDidCompleteWith activationState: WCSessionActivationState,
    error: Error?
  ) {}

  func sessionDidBecomeInactive(_ session: WCSession) {}

  func sessionDidDeactivate(_ session: WCSession) {
    session.activate()
  }

  func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
    handle(message)
  }

  func session(
    _ session: WCSession,
    didReceiveMessage message: [String: Any],
    replyHandler: @escaping ([String: Any]) -> Void
  ) {
    handle(message, replyHandler: replyHandler)
  }

  func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
    handle(userInfo)
  }
}
