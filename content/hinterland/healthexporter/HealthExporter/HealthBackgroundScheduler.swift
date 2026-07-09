import BackgroundTasks
import Foundation

final class HealthBackgroundScheduler {
  static let shared = HealthBackgroundScheduler()
  static let taskIdentifier = "xyz.aarnphm.healthexporter.export"

  private init() {}

  func register() {
    BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.taskIdentifier, using: nil) { task in
      guard let task = task as? BGAppRefreshTask else {
        task.setTaskCompleted(success: false)
        return
      }
      self.handle(task)
    }
  }

  func schedule() {
    let request = BGAppRefreshTaskRequest(identifier: Self.taskIdentifier)
    request.earliestBeginDate = nextRefreshDate()
    try? BGTaskScheduler.shared.submit(request)
  }

  private func handle(_ task: BGAppRefreshTask) {
    schedule()
    let exportTask = Task {
      do {
        _ = try await HealthExportRuntime.shared.export(force: true)
        task.setTaskCompleted(success: true)
      } catch {
        task.setTaskCompleted(success: false)
      }
    }
    task.expirationHandler = {
      exportTask.cancel()
    }
  }

  private func nextRefreshDate() -> Date {
    var calendar = Calendar.autoupdatingCurrent
    calendar.timeZone = .autoupdatingCurrent
    let now = Date()
    var components = calendar.dateComponents([.year, .month, .day], from: now)
    components.hour = 2
    components.minute = 30
    components.second = 0
    let today = calendar.date(from: components) ?? now.addingTimeInterval(6 * 60 * 60)
    if today > now { return today }
    return calendar.date(byAdding: .day, value: 1, to: today) ?? now.addingTimeInterval(24 * 60 * 60)
  }
}
