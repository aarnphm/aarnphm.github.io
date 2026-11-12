import SwiftUI

@available(macOS 14.2, *)
struct AudioOutputApp: App {
    @NSApplicationDelegateAdaptor(RouterAppDelegate.self) private var delegate
    @StateObject private var model = RouterModel()

    var body: some Scene {
        Window("Audio Output", id: "router") {
            RouterView(model: model)
        }.defaultSize(width: 640, height: 620)
        MenuBarExtra {
            RouterView(model: model).frame(width: 560, height: 570)
        } label: {
            Image(nsImage: model.paused ? MenuIcons.paused : MenuIcons.active)
                .accessibilityLabel(model.paused ? "Audio Output, paused" : "Audio Output")
        }.menuBarExtraStyle(.window)
    }
}

private final class RouterAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        guard let window = sender.windows.first(where: { $0.identifier?.rawValue == "router" }) else { return true }
        window.makeKeyAndOrderFront(nil)
        sender.activate(ignoringOtherApps: true)
        return false
    }
}

private enum MenuIcons {
    static let active = load("MenuIcon", fallback: "headphones")
    static let paused = load("MenuIconPaused", fallback: "pause.circle")

    private static func load(_ name: String, fallback: String) -> NSImage {
        let image = NSImage(named: name)
            ?? NSImage(systemSymbolName: fallback, accessibilityDescription: "Audio Output")
            ?? NSImage(size: NSSize(width: 18, height: 18))
        image.size = NSSize(width: 18, height: 18)
        image.isTemplate = true
        return image
    }
}
