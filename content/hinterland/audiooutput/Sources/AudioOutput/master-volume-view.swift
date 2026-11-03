import SwiftUI

@available(macOS 14.2, *)
struct MasterVolumeView: View {
    @ObservedObject var master: OutputMaster
    let outputName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("System output: \(outputName)").foregroundStyle(.secondary)
                Spacer()
                if master.eligible {
                    if master.enabled && master.keyAccess {
                        Text("\(Int((master.level.gain * 100).rounded()))%")
                            .monospacedDigit().accessibilityLabel("Master volume")
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
            if master.eligible && master.enabled && !master.keyAccess {
                Button("Allow volume keys…") { master.requestAccess() }
            }
            if let issue = master.issue {
                Text(issue).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }.font(.callout)
    }
}
