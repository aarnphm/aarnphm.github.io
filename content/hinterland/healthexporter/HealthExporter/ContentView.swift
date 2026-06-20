import SwiftUI

struct ContentView: View {
  @StateObject private var model = HealthExporterViewModel()

  var body: some View {
    NavigationStack {
      Form {
        Section("Status") {
          LabeledContent("State", value: model.status)
          LabeledContent("Last export", value: model.generatedAt)
          LabeledContent("Days", value: "\(model.dayCount)")
          LabeledContent("Swims", value: "\(model.swimCount)")
        }
        Section("Output") {
          Text(model.filePath)
            .font(.footnote)
            .textSelection(.enabled)
        }
      }
      .navigationTitle("HealthExporter")
      .safeAreaInset(edge: .bottom) {
        syncButton
          .buttonStyle(.borderedProminent)
          .padding()
          .background(.regularMaterial)
      }
    }
    .task {
      await model.prepare()
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
