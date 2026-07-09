import SwiftUI

struct ContentView: View {
  @Environment(\.scenePhase) private var scenePhase
  @StateObject private var model = HealthExporterViewModel()
  @State private var isStatusExpanded = false

  var body: some View {
    Form {
      Section("Status") {
        DisclosureGroup(isExpanded: $isStatusExpanded) {
          exportDetails
        } label: {
          LabeledContent("State", value: model.state.rawValue)
        }
        .onChange(of: isStatusExpanded) { _, expanded in
          if expanded {
            Task {
              await model.loadDetails()
            }
          }
        }
      }
      Section("Output") {
        Text(HealthExportWriter.visiblePath)
          .font(.footnote)
          .textSelection(.enabled)
      }
    }
    .safeAreaInset(edge: .bottom) {
      syncButton
        .buttonStyle(.borderedProminent)
        .padding()
        .background(.regularMaterial)
    }
    .task {
      await model.prepare()
    }
    .onChange(of: scenePhase) { _, phase in
      if phase == .active {
        Task {
          model.refreshState()
          if isStatusExpanded {
            await model.loadDetails()
          }
        }
      }
    }
  }

  @ViewBuilder
  private var exportDetails: some View {
    switch model.detailsState {
    case .idle, .loading:
      HStack {
        Text("Loading details")
        Spacer()
        ProgressView()
          .controlSize(.small)
      }
      .foregroundStyle(.secondary)
    case .available(let details):
      LabeledContent("Last export", value: details.generatedAt)
      LabeledContent("Days", value: details.dayCount.formatted())
      LabeledContent("Swims", value: details.swimCount.formatted())
      LabeledContent("Workouts", value: details.workoutCount.formatted())
      LabeledContent("HR samples", value: details.heartRateCount.formatted())
    case .unavailable:
      Text("No export is available yet.")
        .foregroundStyle(.secondary)
    }
  }

  private var syncTitle: String {
    model.isExporting ? "Syncing" : "Sync Now"
  }

  private var syncButton: some View {
    Button {
      model.exportNow()
    } label: {
      Label(syncTitle, systemImage: "arrow.triangle.2.circlepath")
        .frame(maxWidth: .infinity)
    }
    .disabled(model.isExporting)
  }
}
