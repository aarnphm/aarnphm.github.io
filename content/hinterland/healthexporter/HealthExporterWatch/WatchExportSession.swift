import Foundation
import WatchConnectivity

final class WatchExportSession: NSObject, ObservableObject, WCSessionDelegate {
  static let shared = WatchExportSession()

  @Published var status = "Ready"
  @Published var lastExport = "Never"
  @Published var dayCount = 0
  @Published var swimCount = 0
  @Published var workoutCount = 0
  @Published var heartRateCount = 0

  private override init() {
    super.init()
  }

  func activate() {
    guard WCSession.isSupported() else {
      status = "Watch Connectivity unavailable"
      return
    }
    let session = WCSession.default
    session.delegate = self
    session.activate()
  }

  func requestExport() {
    guard WCSession.isSupported() else {
      status = "Watch Connectivity unavailable"
      return
    }
    status = "Requesting export"
    let message: [String: Any] = ["command": "export", "requestedAt": Date().timeIntervalSince1970]
    let session = WCSession.default
    if session.isReachable {
      session.sendMessage(
        message,
        replyHandler: { reply in
          DispatchQueue.main.async {
            self.status = (reply["accepted"] as? Bool) == true ? "Export requested" : "Phone rejected request"
          }
        },
        errorHandler: { error in
          DispatchQueue.main.async {
            self.status = error.localizedDescription
          }
        }
      )
    } else {
      session.transferUserInfo(message)
      status = "Queued for phone"
    }
  }

  func session(
    _ session: WCSession,
    activationDidCompleteWith activationState: WCSessionActivationState,
    error: Error?
  ) {
    DispatchQueue.main.async {
      self.status = error?.localizedDescription ?? "Ready"
    }
  }

  func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
    update(applicationContext)
  }

  private func update(_ context: [String: Any]) {
    DispatchQueue.main.async {
      if let generatedAt = context["generatedAt"] as? String {
        self.lastExport = generatedAt
      }
      if let dayCount = context["dayCount"] as? Int {
        self.dayCount = dayCount
      }
      if let swimCount = context["swimCount"] as? Int {
        self.swimCount = swimCount
      }
      if let workoutCount = context["workoutCount"] as? Int {
        self.workoutCount = workoutCount
      }
      if let heartRateCount = context["heartRateCount"] as? Int {
        self.heartRateCount = heartRateCount
      }
      self.status = "Synced from phone"
    }
  }
}
