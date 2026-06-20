import SwiftUI

struct WatchContentView: View {
  @ObservedObject var session: WatchExportSession

  var body: some View {
    List {
      Section {
        Text(session.status)
        LabeledContent("Last", value: session.lastExport)
        LabeledContent("Days", value: "\(session.dayCount)")
        LabeledContent("Swims", value: "\(session.swimCount)")
      }
      Section {
        Button("Sync Now") {
          session.requestExport()
        }
      }
    }
    .navigationTitle("HealthExporter")
  }
}
