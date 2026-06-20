import SwiftUI
import WatchKit

@main
struct HealthExporterWatchApp: App {
  @WKExtensionDelegateAdaptor(WatchExtensionDelegate.self) private var delegate

  var body: some Scene {
    WindowGroup {
      NavigationStack {
        WatchContentView(session: .shared)
      }
    }
  }
}

final class WatchExtensionDelegate: NSObject, WKExtensionDelegate {
  func applicationDidFinishLaunching() {
    WatchExportSession.shared.activate()
    WatchBackgroundScheduler.shared.schedule()
  }

  func applicationDidEnterBackground() {
    WatchBackgroundScheduler.shared.schedule()
  }

  func handle(_ backgroundTasks: Set<WKRefreshBackgroundTask>) {
    WatchBackgroundScheduler.shared.handle(backgroundTasks)
  }
}
