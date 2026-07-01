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
        LabeledContent("Workouts", value: "\(session.workoutCount)")
        LabeledContent("HR", value: "\(session.heartRateCount)")
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
