import SwiftUI

struct TetherAccountSheet: View {
  @ObservedObject var model: TetherModel
  @Environment(\.dismiss) private var dismiss
  @Environment(\.colorScheme) private var scheme

  var body: some View {
    let theme = GardenTheme(scheme: scheme)
    VStack(alignment: .leading, spacing: 24) {
      HStack {
        Text("garmin connect").font(.system(size: 22, weight: .medium))
        Spacer()
        Button("done") { dismiss() }.keyboardShortcut(.cancelAction)
          .buttonStyle(GardenButtonStyle())
          .focusEffectDisabled()
      }
      HStack(spacing: 12) {
        if let connectURL = URL(string: "https://connect.garmin.com/modern/") {
          Link("login", destination: connectURL)
            .buttonStyle(GardenButtonStyle(prominent: true))
            .focusEffectDisabled()
            .help("Open Garmin Connect in browser")
        }
        if model.accountStatus == "keychain unavailable"
          || model.accountStatus == "waiting for keychain"
        {
          Button("unlock keychain", action: model.authorizeAccount)
            .buttonStyle(GardenButtonStyle())
            .focusEffectDisabled()
            .disabled(model.accountStatus == "waiting for keychain")
        }
      }
    }
    .padding(32).frame(width: 490)
    .font(GardenTheme.body).foregroundStyle(theme.ink).background(theme.paper)
  }
}

struct TetherAddDeviceSheet: View {
  @ObservedObject var model: TetherModel
  @Environment(\.dismiss) private var dismiss
  @Environment(\.colorScheme) private var scheme

  var body: some View {
    let theme = GardenTheme(scheme: scheme)
    VStack(alignment: .leading, spacing: 24) {
      HStack {
        Text("add a device").font(.system(size: 22, weight: .medium))
        Spacer()
        Button("done") { dismiss() }.keyboardShortcut(.cancelAction)
          .buttonStyle(GardenButtonStyle())
      }
      VStack(alignment: .leading, spacing: 12) {
        Image(systemName: "cable.connector").font(.system(size: 36, weight: .light))
        Text("Connect your Garmin with a USB data cable.")
        Text("If prompted, choose file transfer mode.")
          .foregroundStyle(theme.muted).fixedSize(horizontal: false, vertical: true)
      }
      if model.devices.contains(where: \.connected) {
        VStack(spacing: 0) {
          ForEach(model.devices.filter(\.connected)) { device in
            Button {
              model.selectedDeviceID = device.id
              dismiss()
            } label: {
              HStack {
                Image(systemName: "applewatch")
                Text(device.name)
                Spacer()
                Text("select").font(GardenTheme.meta)
              }.frame(maxWidth: .infinity, alignment: .leading)
            }.buttonStyle(GardenButtonStyle()).padding(.bottom, 8)
          }
        }
      } else {
        Label("waiting for a device", systemImage: "cable.connector")
          .font(GardenTheme.meta).padding(20)
          .frame(maxWidth: .infinity, alignment: .leading)
          .background(theme.surface)
          .overlay(Rectangle().strokeBorder(theme.line, lineWidth: 1))
      }
      HStack {
        Spacer()
        Button("scan again", action: model.reconnect).buttonStyle(GardenButtonStyle())
      }
    }
    .padding(32).frame(width: 490)
    .font(GardenTheme.body).foregroundStyle(theme.ink).background(theme.paper)
  }
}

struct TetherAboutView: View {
  @Environment(\.colorScheme) private var scheme

  var body: some View {
    let theme = GardenTheme(scheme: scheme)
    VStack(spacing: 16) {
      Image(nsImage: NSApplication.shared.applicationIconImage)
        .resizable().frame(width: 80, height: 80)
      Text("Tether").font(.system(size: 24, weight: .medium))
      Text("0.1.0").font(GardenTheme.meta).foregroundStyle(theme.muted)
      Text("FIT activity transfer and Garmin Connect upload.")
      Text("Firmware and map updates require Garmin Express.")
        .foregroundStyle(theme.muted)
    }
    .font(GardenTheme.body).padding(32).frame(width: 420)
    .foregroundStyle(theme.ink).background(theme.paper)
  }
}
