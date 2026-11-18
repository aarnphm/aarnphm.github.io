public struct SyncProgress: Equatable, Sendable {
  public enum Phase: String, Sendable {
    case connecting, reading, checking, downloading, uploading, verifying
    case complete, cancelling, cancelled, failed
  }

  public var phase: Phase
  public var completed: Int
  public var total: Int?

  public init(phase: Phase, completed: Int = 0, total: Int? = nil) {
    self.phase = phase
    self.completed = completed
    self.total = total
  }

  public var fraction: Double? {
    guard let total else { return nil }
    guard total > 0 else { return phase == .complete ? 1 : 0 }
    return min(1, max(0, Double(completed) / Double(total)))
  }

  public var label: String {
    switch phase {
    case .connecting: "connecting"
    case .reading: "reading activities"
    case .checking: "checking activities"
    case .downloading: "saving activity"
    case .uploading: "uploading to connect"
    case .verifying: "verifying in connect"
    case .complete: "checked"
    case .cancelling: "cancelling"
    case .cancelled: "cancelled"
    case .failed: "stopped"
    }
  }
}
