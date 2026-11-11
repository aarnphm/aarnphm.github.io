import SwiftUI

@main
struct TetherApp: App {
  @StateObject private var model = TetherModel()
  @AppStorage("appearance") private var appearance = "system"
  @Environment(\.openWindow) private var openWindow

  private var colorScheme: ColorScheme? {
    switch appearance {
    case "light": .light
    case "dark": .dark
    default: nil
    }
  }

  var body: some Scene {
    WindowGroup("Tether") {
      TetherView(model: model)
        .frame(minWidth: 780, minHeight: 560)
        .preferredColorScheme(colorScheme)
        .task { model.start() }
        .onReceive(
          NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
        ) { _ in
          model.refreshConnection()
        }
        .onReceive(
          NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didWakeNotification)
        ) { _ in
          model.refreshConnection()
        }
    }
    .defaultSize(width: 1040, height: 720)
    .windowStyle(.hiddenTitleBar)
    .commands {
      CommandGroup(replacing: .appInfo) {
        Button("About Tether") { openWindow(id: "about") }
      }
      CommandGroup(after: .toolbar) {
        Picker("Appearance", selection: $appearance) {
          Text("System").tag("system")
          Text("Light").tag("light")
          Text("Dark").tag("dark")
        }
      }
      CommandGroup(replacing: .newItem) {
        Button("Import FIT files…", action: model.importActivities)
          .keyboardShortcut("o", modifiers: .command)
      }
      CommandMenu("Device") {
        Button("Sync activities", action: model.sync)
          .keyboardShortcut("r", modifiers: .command)
          .disabled(
            model.isSyncing || model.selectedDevice?.connected != true
              || model.selectedDevice?.busyOwner != nil)
        Button("Add device…") { model.showAddDevice = true }
        Button("Reconnect", action: model.reconnect)
        Divider()
        Button("Account…") { model.showAccount = true }
        Button("Show activity archive", action: model.revealArchive)
      }
    }
    Window("About Tether", id: "about") {
      TetherAboutView().preferredColorScheme(colorScheme)
    }
    .windowResizability(.contentSize)
  }
}
