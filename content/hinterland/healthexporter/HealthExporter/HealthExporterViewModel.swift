import Foundation

struct HealthExportDetails {
  let generatedAt: String
  let dayCount: Int
  let swimCount: Int
  let workoutCount: Int
  let heartRateCount: Int

  init(result: HealthExportResult) {
    generatedAt = result.document.generatedAt
    dayCount = result.document.days.count
    swimCount = result.document.swims.count
    workoutCount = result.document.workouts.count
    heartRateCount = result.document.workouts.reduce(0) { $0 + $1.heartRate.count }
  }
}

enum HealthExportDetailsState {
  case idle
  case loading
  case available(HealthExportDetails)
  case unavailable
}

@MainActor
final class HealthExporterViewModel: ObservableObject {
  private static let catchUpInterval: TimeInterval = 60 * 60
  private let stateStore: HealthExportStateStore

  @Published private(set) var state: HealthExportState
  @Published private(set) var detailsState = HealthExportDetailsState.idle

  private var detailsRequested = false

  var isExporting: Bool {
    state == .exporting
  }

  init(stateStore: HealthExportStateStore = HealthExportStateStore()) {
    self.stateStore = stateStore
    state = stateStore.state(maxAge: Self.catchUpInterval)
  }

  func refreshState() {
    guard !isExporting else { return }
    state = stateStore.state(maxAge: Self.catchUpInterval)
  }

  func prepare() async {
    do {
      try await HealthExportRuntime.shared.requestAuthorization()
      HealthBackgroundScheduler.shared.schedule()
      guard stateStore.state(maxAge: Self.catchUpInterval) == .stale else {
        state = .exported
        return
      }
      state = .exporting
      try await HealthExportRuntime.shared.catchUpIfNeeded(maxAge: Self.catchUpInterval)
      state = .exported
      if detailsRequested {
        await loadDetails()
      }
    } catch {
      state = .failed
    }
  }

  func loadDetails() async {
    detailsRequested = true
    if case .loading = detailsState { return }
    detailsState = .loading
    guard let result = await HealthExportRuntime.shared.currentExport() else {
      detailsState = .unavailable
      return
    }
    detailsState = .available(HealthExportDetails(result: result))
  }

  func exportNow() {
    guard !isExporting else { return }
    state = .exporting
    Task {
      do {
        let result = try await HealthExportRuntime.shared.export(force: true)
        state = .exported
        if detailsRequested {
          detailsState = .available(HealthExportDetails(result: result))
        }
      } catch {
        state = .failed
      }
    }
  }
}
