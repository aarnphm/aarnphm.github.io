import Foundation
import WatchKit

final class WatchBackgroundScheduler {
  static let shared = WatchBackgroundScheduler()

  private init() {}

  func schedule() {
    WKExtension.shared().scheduleBackgroundRefresh(
      withPreferredDate: nextRefreshDate(),
      userInfo: nil
    ) { _ in }
  }

  func handle(_ backgroundTasks: Set<WKRefreshBackgroundTask>) {
    for task in backgroundTasks {
      if task is WKApplicationRefreshBackgroundTask {
        WatchExportSession.shared.requestExport()
        schedule()
      }
      task.setTaskCompletedWithSnapshot(false)
    }
  }

  private func nextRefreshDate() -> Date {
    var calendar = Calendar.autoupdatingCurrent
    calendar.timeZone = .autoupdatingCurrent
    let now = Date()
    var components = calendar.dateComponents([.year, .month, .day], from: now)
    components.hour = 2
    components.minute = 20
    components.second = 0
    let today = calendar.date(from: components) ?? now.addingTimeInterval(6 * 60 * 60)
    if today > now { return today }
    return calendar.date(byAdding: .day, value: 1, to: today) ?? now.addingTimeInterval(24 * 60 * 60)
  }
}
