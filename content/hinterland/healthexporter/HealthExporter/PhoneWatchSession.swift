import Foundation
import WatchConnectivity

final class PhoneWatchSession: NSObject, WCSessionDelegate, @unchecked Sendable {
  static let shared = PhoneWatchSession()

  private let exportHandlerLock = NSLock()
  private var exportHandler: (@Sendable () -> Void)?

  private override init() {
    super.init()
  }

  func start(exportHandler: @escaping @Sendable () -> Void) {
    guard WCSession.isSupported() else { return }
    exportHandlerLock.lock()
    self.exportHandler = exportHandler
    exportHandlerLock.unlock()
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
      "workoutCount": result.document.workouts.count,
      "heartRateCount": result.document.workouts.reduce(0) { $0 + $1.heartRate.count },
      "routeCount": result.document.workouts.filter { $0.gpxFile != nil }.count,
      "path": result.url.path,
    ]
    try? WCSession.default.updateApplicationContext(context)
  }

  private func handle(_ message: [String: Any], replyHandler: (([String: Any]) -> Void)? = nil) {
    guard message["command"] as? String == "export" else {
      replyHandler?(["accepted": false])
      return
    }
    exportHandlerLock.lock()
    let handler = exportHandler
    exportHandlerLock.unlock()
    handler?()
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
