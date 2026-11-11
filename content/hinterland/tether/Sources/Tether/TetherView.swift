import SwiftUI
import TetherCore

struct TetherView: View {
  @ObservedObject var model: TetherModel
  @FocusState private var addDeviceFocused: Bool
  @Environment(\.colorScheme) private var scheme
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  private var theme: GardenTheme { GardenTheme(scheme: scheme) }

  var body: some View {
    HStack(spacing: 0) {
      sidebar.frame(width: 224)
      Rectangle().fill(theme.line).frame(width: 1).ignoresSafeArea(.container, edges: .top)
      VStack(alignment: .leading, spacing: 0) {
        ScrollView {
          VStack(alignment: .leading, spacing: 28) {
            header
            if let progress = model.syncProgress {
              TetherProgressView(progress: progress).transition(.opacity)
            }
            if let error = model.error { errorMessage(error) }
            if model.selectedDevice == nil && !model.isSyncing && model.syncProgress == nil {
              emptyDevice
            }
            activityList
          }
          .padding(32)
          .frame(maxWidth: .infinity, alignment: .leading)
          .animation(
            reduceMotion ? nil : .easeOut(duration: 0.22), value: model.syncProgress != nil)
        }
      }
    }
    .font(GardenTheme.body)
    .foregroundStyle(theme.ink)
    .background(theme.paper.ignoresSafeArea())
    .background(TetherWindowSurface(color: theme.paper))
    .sheet(isPresented: $model.showAccount) { TetherAccountSheet(model: model) }
    .sheet(isPresented: $model.showAddDevice) { TetherAddDeviceSheet(model: model) }
  }

  private var sidebar: some View {
    VStack(alignment: .leading, spacing: 0) {
      Text("tether").font(.system(size: 20, weight: .medium)).padding(24)
      HStack {
        Text("devices").font(GardenTheme.meta).foregroundStyle(theme.muted)
        Spacer()
        Text("\(model.devices.count)").font(GardenTheme.meta).monospacedDigit()
      }.padding(.horizontal, 24).padding(.bottom, 12)
      ScrollView {
        VStack(spacing: 1) {
          ForEach(model.devices) { device in
            Button {
              model.selectedDeviceID = device.id
            } label: {
              HStack(spacing: 12) {
                Image(systemName: "applewatch").font(.system(size: 22))
                VStack(alignment: .leading, spacing: 5) {
                  Text(device.name).font(GardenTheme.body).lineLimit(2)
                  Text(
                    device.busyOwner.map { "in use by \($0)" }
                      ?? (device.connected ? "connected" : "disconnected")
                  )
                  .font(GardenTheme.meta)
                }
                Spacer(minLength: 0)
              }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .background(model.selectedDeviceID == device.id ? theme.selected : theme.surface)
            .accessibilityAddTraits(model.selectedDeviceID == device.id ? .isSelected : [])
            .padding(.horizontal, 8)
          }
          if model.devices.isEmpty {
            Text("No saved devices.")
              .font(GardenTheme.meta).foregroundStyle(theme.muted)
              .frame(maxWidth: .infinity, alignment: .leading).padding(24)
          }
        }
      }
      Button {
        model.showAddDevice = true
      } label: {
        Label("add device", systemImage: "plus").frame(maxWidth: .infinity)
      }
      .buttonStyle(GardenButtonStyle())
      .focused($addDeviceFocused)
      .focusEffectDisabled()
      .overlay {
        if addDeviceFocused {
          Rectangle().strokeBorder(theme.ink, lineWidth: 2).allowsHitTesting(false)
        }
      }
      .padding(16)
    }.background(theme.surface.ignoresSafeArea(.container, edges: .top))
  }

  private var header: some View {
    HStack(alignment: .center, spacing: 12) {
      Text(model.selectedDevice?.name ?? "devices")
        .font(.system(size: 24, weight: .medium))
        .lineLimit(1)
        .truncationMode(.tail)
        .help(model.selectedDevice?.name ?? "devices")
      Spacer(minLength: 0)
      HStack(spacing: 8) {
        connectionControl
        Button {
          if model.isSyncing { model.cancelSync() } else { model.sync() }
        } label: {
          Image(systemName: model.isSyncing ? "stop" : "arrow.triangle.2.circlepath")
            .frame(width: 30, height: 30)
        }
        .disabled(!model.isSyncing && !deviceReady)
        .accessibilityLabel(model.isSyncing ? "Cancel transfer" : "Sync activities")
        .help(model.isSyncing ? "Cancel transfer" : "Sync activities")
        Menu {
          if let device = model.selectedDevice {
            Text("Last seen \(device.lastSeen.formatted(date: .abbreviated, time: .shortened))")
            Button("Forget this device", action: model.forgetSelectedDevice)
              .disabled(model.isSyncing)
            Divider()
          }
          Button("Account settings…") { model.showAccount = true }
        } label: {
          Image(systemName: "ellipsis")
            .frame(width: 30, height: 30)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 30, height: 30)
        .accessibilityLabel("Device options")
        .help("Device options")
        Button(action: model.importActivities) {
          Label("import FIT", systemImage: "square.and.arrow.down").frame(height: 30)
        }
        .disabled(model.isSyncing)
        .help("Import activity files")
        if let connectURL = URL(string: "https://connect.garmin.com/modern/") {
          Link(destination: connectURL) {
            Image(systemName: "person.crop.circle").frame(width: 30, height: 30)
          }
          .accessibilityLabel("Open Garmin Connect")
          .help("Open Garmin Connect in browser")
        }
      }
      .font(GardenTheme.meta)
      .buttonStyle(.borderless)
      .fixedSize()
    }
  }

  private var deviceReady: Bool {
    model.selectedDevice?.connected == true && model.selectedDevice?.busyOwner == nil
  }

  @ViewBuilder private var connectionControl: some View {
    if deviceReady {
      Image(systemName: "cable.connector")
        .frame(width: 30, height: 30)
        .accessibilityLabel("Watch connected over USB")
        .help("Watch connected over USB")
    } else {
      Button(action: model.reconnect) {
        Image(
          systemName: model.selectedDevice?.connected == true
            ? "exclamationmark.triangle" : "cable.connector.slash"
        )
        .frame(width: 30, height: 30)
      }
      .disabled(model.isSyncing)
      .accessibilityLabel(connectionHelp)
      .help(connectionHelp)
    }
  }

  private var connectionHelp: String {
    if let owner = model.selectedDevice?.busyOwner { return "Watch in use by \(owner). Reconnect" }
    return "Watch disconnected. Reconnect"
  }

  private var emptyDevice: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Connect a Garmin with a USB data cable.").foregroundStyle(theme.muted)
      Button("add device") { model.showAddDevice = true }
        .buttonStyle(GardenButtonStyle(prominent: true))
    }.frame(maxWidth: .infinity, alignment: .leading)
  }

  private var activityList: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        Text("activities").font(.system(size: 16, weight: .medium))
        Spacer()
        Button(action: model.revealArchive) {
          Image(systemName: "folder").frame(width: 30, height: 30)
        }
        .buttonStyle(.borderless)
        .accessibilityLabel("Show activity archive")
        .help("Show activity archive")
      }
      if model.selectedRecords.isEmpty {
        VStack(alignment: .leading, spacing: 8) {
          Text("no transfers yet").font(GardenTheme.meta)
        }.padding(.vertical, 24)
          .frame(maxWidth: .infinity, alignment: .leading)
          .overlay(alignment: .top) { Rectangle().fill(theme.line).frame(height: 1) }
      } else {
        LazyVStack(spacing: 0) {
          ForEach(model.selectedRecords) { record in
            TetherTransferRow(record: record)
            Rectangle().fill(theme.line).frame(height: 1)
          }
        }
      }
    }
  }

  private func errorMessage(_ error: String) -> some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: "exclamationmark.triangle")
      Text(error).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
      Button("reconnect", action: model.reconnect).buttonStyle(GardenButtonStyle())
    }.padding(16).background(theme.selected)
  }

}

private struct TetherTransferRow: View {
  let record: TransferRecord
  @Environment(\.colorScheme) private var scheme
  var body: some View {
    let theme = GardenTheme(scheme: scheme)
    HStack(spacing: 16) {
      Image(systemName: "doc").foregroundStyle(theme.muted)
      VStack(alignment: .leading, spacing: 6) {
        Text(record.filename).font(GardenTheme.body).lineLimit(1).truncationMode(.middle)
        Text(dateLabel)
          .font(GardenTheme.meta).foregroundStyle(theme.muted)
          .help(
            record.sourceModified == nil
              ? "Date saved to the local archive" : "File modification date on the watch")
      }
      Spacer(minLength: 12)
      VStack(alignment: .trailing, spacing: 6) {
        if record.state == "confirmed" {
          Image(systemName: "checkmark")
            .foregroundStyle(theme.ink)
            .accessibilityLabel("Confirmed in Garmin Connect")
            .help("Confirmed in Garmin Connect")
        } else {
          Text(record.state).foregroundStyle(theme.ink)
        }
        Text(ByteCountFormatter.string(fromByteCount: record.bytes, countStyle: .file))
          .foregroundStyle(theme.muted)
      }.font(GardenTheme.meta).monospacedDigit()
      if let url = activityURL {
        Link(destination: url) {
          Image(systemName: "arrow.up.right").frame(width: 24, height: 24)
        }.accessibilityLabel("Open activity in Garmin Connect").help("Open in Garmin Connect")
      }
    }
    .padding(.vertical, 14)
    .contextMenu {
      if !record.localPath.isEmpty {
        Button("Show in Finder") {
          NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: record.localPath)])
        }
      }
      if let url = activityURL { Link("Open in Garmin Connect", destination: url) }
    }
    .accessibilityElement(children: .contain)
  }

  private var dateLabel: String {
    if let modified = record.sourceModified {
      return modified.formatted(date: .abbreviated, time: .omitted)
    }
    return "saved · \(record.date.formatted(date: .abbreviated, time: .omitted))"
  }

  private var activityURL: URL? {
    guard let id = record.activityID, !id.isEmpty,
      let base = URL(string: "https://connect.garmin.com/modern/activity")
    else { return nil }
    return base.appendingPathComponent(id)
  }

}
