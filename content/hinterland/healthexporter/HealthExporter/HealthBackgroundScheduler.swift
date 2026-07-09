import BackgroundTasks
import Foundation

private final class BackgroundRefreshCompletion: @unchecked Sendable {
  private let task: BGAppRefreshTask
  private let lock = NSLock()
  private var isCompleted = false

  init(_ task: BGAppRefreshTask) {
    self.task = task
  }

  func complete(success: Bool) {
    lock.lock()
    guard !isCompleted else {
      lock.unlock()
      return
    }
    isCompleted = true
    lock.unlock()
    task.setTaskCompleted(success: success)
  }
}

final class HealthBackgroundScheduler: @unchecked Sendable {
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
    BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: Self.taskIdentifier)
    let request = BGAppRefreshTaskRequest(identifier: Self.taskIdentifier)
    request.earliestBeginDate = nextRefreshDate()
    try? BGTaskScheduler.shared.submit(request)
  }

  private func handle(_ task: BGAppRefreshTask) {
    schedule()
    let completion = BackgroundRefreshCompletion(task)
    let exportTask = Task {
      do {
        _ = try await HealthExportRuntime.shared.export(force: true)
        completion.complete(success: true)
      } catch {
        completion.complete(success: false)
      }
    }
    task.expirationHandler = {
      exportTask.cancel()
      completion.complete(success: false)
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
