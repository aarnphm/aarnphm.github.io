import SwiftUI

@available(macOS 14.2, *)
struct AudioOutputApp: App {
    @StateObject private var model = RouterModel()

    var body: some Scene {
        MenuBarExtra("Audio Output", systemImage: model.paused ? "speaker.slash" : "hifispeaker") {
            RouterView(model: model).frame(width: 560, height: 570)
        }.menuBarExtraStyle(.window)
        Window("Audio Output", id: "router") {
            RouterView(model: model)
        }.defaultSize(width: 640, height: 620)
    }
}
