import SwiftUI

@available(macOS 14.2, *)
struct RouterView: View {
    @ObservedObject var model: RouterModel
    @Environment(\.openWindow) private var openWindow
    @State private var search = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Audio Output").font(.title2.weight(.semibold))
                    Text(model.paused ? "Routing paused" : "\(model.activeCount) active routes · \(model.outputs.count) outputs")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Button(model.paused ? "Resume" : "Pause", systemImage: model.paused ? "play.fill" : "pause.fill") {
                    model.togglePause()
                }
            }
            Text("System output: \(model.defaultOutputName)").font(.callout).foregroundStyle(.secondary)
            if let issue = model.issue {
                Label(issue, systemImage: "exclamationmark.triangle").font(.callout).textSelection(.enabled)
            }
            if model.outputs.isEmpty {
                ContentUnavailableView("No outputs connected", systemImage: "speaker.slash",
                                       description: Text("Connect your speakers or Bluetooth headphones to begin."))
            } else {
                TextField("Find an application", text: $search).textFieldStyle(.roundedBorder)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(model.siteRules.filter { search.isEmpty || $0.name.localizedCaseInsensitiveContains(search) || $0.id.localizedCaseInsensitiveContains(search) }) { site in
                            siteRow(site)
                            Divider()
                        }
                        if model.applications.isEmpty {
                            Text("Play audio in an application to make it appear here.")
                                .foregroundStyle(.secondary).padding(.vertical, 24)
                        }
                        ForEach(model.applications.filter { $0.siteHost == nil && (search.isEmpty || $0.name.localizedCaseInsensitiveContains(search) || $0.id.localizedCaseInsensitiveContains(search)) }) { app in
                            applicationRow(app)
                            Divider()
                        }
                        ForEach(model.assignments.filter { saved in !model.applications.contains { $0.id == saved.id } }) { saved in
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(saved.name)
                                    Text("\(saved.deviceName) · \(model.statuses[saved.id] ?? "Waiting")")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Button("Remove", systemImage: "xmark") { model.forget(saved.id) }
                                    .labelStyle(.iconOnly).help("Remove saved assignment")
                            }.padding(.vertical, 12)
                            Divider()
                        }
                    }
                }.frame(minHeight: 180, idealHeight: 320, maxHeight: .infinity)
            }
            Divider()
            HStack {
                Button("Retry routes", systemImage: "arrow.clockwise") { model.retry() }
                Spacer()
                Menu {
                    Button("Open routing window") { openWindow(id: "router"); NSApp.activate(ignoringOtherApps: true) }
                    Toggle("Open at login", isOn: Binding(get: { model.launchAtLogin }, set: model.setLogin))
                    Button("Browser extension…") {
                        let folder = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/Extension")
                        NSWorkspace.shared.open(folder)
                    }
                    Button("Audio capture permission…") {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    Divider()
                    Button("Quit Audio Output", role: .destructive) {
                        model.shutdown()
                        NSApplication.shared.terminate(nil)
                    }.keyboardShortcut("q")
                } label: { Image(systemName: "gearshape") }.menuStyle(.borderlessButton).fixedSize()
                .accessibilityLabel("Audio Output settings")
            }
        }
        .padding(22)
        .frame(minWidth: 510, minHeight: 460)
    }

    private func siteRow(_ site: SiteRule) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "globe").frame(width: 30, height: 30)
            VStack(alignment: .leading, spacing: 4) {
                Text(site.name).fontWeight(.medium)
                Text(model.siteStatuses[site.id] ?? "Waiting for browser extension")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }.frame(maxWidth: .infinity, alignment: .leading).help(site.id)
            Picker("Output for \(site.name)", selection: Binding(
                get: { model.siteRules.first { $0.id == site.id }?.outputUID ?? "" },
                set: { model.assignSite(site, uid: $0) }
            )) {
                Text("Normal output").tag("")
                ForEach(model.outputs) { device in Text(device.name).tag(device.uid) }
                if !site.outputUID.isEmpty && !model.outputs.contains(where: { $0.uid == site.outputUID }) {
                    Text("\(site.outputName) (disconnected)").tag(site.outputUID)
                }
            }.labelsHidden().frame(width: 210)
        }.padding(.vertical, 12)
    }

    private func applicationRow(_ app: AudioApplication) -> some View {
        HStack(spacing: 12) {
            if let icon = app.icon { Image(nsImage: icon).resizable().frame(width: 30, height: 30) }
            else { Image(systemName: "app.dashed").frame(width: 30, height: 30) }
            VStack(alignment: .leading, spacing: 4) {
                Text(app.name).fontWeight(.medium)
                Text(app.sharedAudioOwner.map { "Shares \($0) audio" } ?? model.statuses[app.id] ?? "Normal output").font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }.frame(maxWidth: .infinity, alignment: .leading).help(app.id)
            Picker("Output for \(app.name)", selection: Binding(
                get: { model.assignments.first { $0.id == app.id }?.deviceUID ?? "" },
                set: { model.assign(app, uid: $0) }
            )) {
                Text("Normal output").tag("")
                ForEach(model.outputs) { device in Text(device.name).tag(device.uid) }
                if let saved = model.assignments.first(where: { $0.id == app.id }),
                   !model.outputs.contains(where: { $0.uid == saved.deviceUID }) {
                    Text("\(saved.deviceName) (disconnected)").tag(saved.deviceUID)
                }
            }.labelsHidden().frame(width: 210).disabled(app.sharedAudioOwner != nil)
        }.padding(.vertical, 12)
    }
}
