import SwiftUI

struct GardenTheme {
  let scheme: ColorScheme
  private func color(_ light: UInt32, _ dark: UInt32) -> Color {
    let value = scheme == .dark ? dark : light
    return Color(
      red: Double((value >> 16) & 255) / 255,
      green: Double((value >> 8) & 255) / 255,
      blue: Double(value & 255) / 255)
  }
  var paper: Color { color(0xfffcf0, 0x100f0f) }
  var ink: Color { color(0x100f0f, 0xcecdc3) }
  var muted: Color { color(0x6f6e69, 0x9f9d96) }
  var surface: Color { color(0xfcf9ed, 0x131212) }
  var selected: Color { color(0xf4f3de, 0x34352a) }
  var line: Color { color(0xb7b5ac, 0x575653).opacity(0.55) }
  var sage: Color { color(0xcdd597, 0xcdd597) }
  var accentInk: Color { color(0x100f0f, 0x100f0f) }
  static let meta = Font.system(size: 12, design: .monospaced)
  static let body = Font.system(size: 14)
}

struct GardenButtonStyle: ButtonStyle {
  var prominent = false
  func makeBody(configuration: Configuration) -> some View {
    GardenButtonBody(configuration: configuration, prominent: prominent)
  }
}

private struct GardenButtonBody: View {
  let configuration: ButtonStyle.Configuration
  let prominent: Bool
  @Environment(\.colorScheme) private var scheme
  @Environment(\.isEnabled) private var enabled
  @Environment(\.isFocused) private var focused
  @State private var hovering = false

  var body: some View {
    let theme = GardenTheme(scheme: scheme)
    configuration.label
      .font(GardenTheme.meta)
      .padding(.horizontal, 12)
      .frame(minHeight: 30)
      .foregroundStyle(prominent ? theme.accentInk : theme.ink)
      .background(
        prominent
          ? theme.sage : (hovering || configuration.isPressed ? theme.selected : theme.paper)
      )
      .overlay(
        Rectangle().strokeBorder(focused ? theme.ink : theme.muted, lineWidth: focused ? 2 : 1)
      )
      .opacity(enabled ? 1 : 0.45)
      .contentShape(Rectangle())
      .onHover { hovering = $0 }
  }
}

struct TetherWindowSurface: NSViewRepresentable {
  let color: Color

  func makeNSView(context: Context) -> TetherWindowBackground {
    TetherWindowBackground(color: NSColor(color))
  }

  func updateNSView(_ view: TetherWindowBackground, context: Context) {
    view.paper = NSColor(color)
    view.apply()
  }
}

final class TetherWindowBackground: NSView {
  var paper: NSColor

  init(color: NSColor) {
    paper = color
    super.init(frame: .zero)
  }

  required init?(coder: NSCoder) { nil }

  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    apply()
  }

  func apply() {
    guard let window else { return }
    window.titlebarAppearsTransparent = true
    window.titleVisibility = .hidden
    window.styleMask.insert(.fullSizeContentView)
    window.backgroundColor = paper
  }
}
