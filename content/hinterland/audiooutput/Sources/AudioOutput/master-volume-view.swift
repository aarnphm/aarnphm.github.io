import SwiftUI

@available(macOS 14.2, *)
struct MasterVolumeView: View {
    @ObservedObject var master: OutputMaster
    let outputName: String
    let outputs: [OutputDevice]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("System output: \(outputName)").foregroundStyle(.secondary)
                Spacer()
                if master.eligible {
                    if master.enabled && master.keyAccess {
                        Text("\(Int((master.level.gain * 100).rounded()))%")
                            .monospacedDigit().accessibilityLabel("Master volume")
                            .accessibilityValue("\(Int((master.level.gain * 100).rounded())) percent")
                    }
                    Menu {
                        Toggle("Use volume keys", isOn: Binding(get: { master.enabled }, set: master.setEnabled))
                        if master.enabled && !master.keyAccess {
                            Button("Allow volume keys…") { master.requestAccess() }
                        }
                    } label: { Image(systemName: "keyboard") }
                    .menuStyle(.borderlessButton).fixedSize()
                    .accessibilityLabel("Volume key settings")
                }
            }
            Picker("Volume-key target", selection: Binding(
                get: { master.targetUID },
                set: { uid in master.selectTarget(outputs.first { $0.uid == uid }) }
            )) {
                Text("System output").tag("")
                ForEach(outputs) { output in Text(output.name).tag(output.uid) }
                if !master.targetUID.isEmpty && !outputs.contains(where: { $0.uid == master.targetUID }) {
                    Text("\(master.targetName) (disconnected)").tag(master.targetUID)
                }
            }
            if !master.targetUID.isEmpty && !master.eligible {
                Text(outputs.contains(where: { $0.uid == master.targetUID })
                     ? "Volume keys require a single stereo output stream."
                     : "Volume-key target disconnected. Reconnect it to resume control.")
                    .foregroundStyle(.secondary)
            }
            if master.eligible && master.enabled && !master.keyAccess {
                Button("Allow volume keys…") { master.requestAccess() }
            }
            if let issue = master.issue {
                Text(issue).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }.font(.callout)
    }
}
