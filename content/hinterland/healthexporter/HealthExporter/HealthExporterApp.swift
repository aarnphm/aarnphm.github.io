import SwiftUI
import UIKit

@main
struct HealthExporterApp: App {
  @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

  var body: some Scene {
    WindowGroup {
      ContentView()
    }
  }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {
    HealthBackgroundScheduler.shared.register()
    HealthBackgroundScheduler.shared.schedule()
    PhoneWatchSession.shared.start {
      Task {
        _ = try? await HealthExportRuntime.shared.export(force: true)
      }
    }
    return true
  }

  func applicationDidEnterBackground(_ application: UIApplication) {
    HealthBackgroundScheduler.shared.schedule()
  }
}
