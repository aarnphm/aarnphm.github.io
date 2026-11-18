import SwiftUI
import TetherCore

struct TetherProgressView: View {
  let progress: SyncProgress?
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    TetherProgressContent(progress: progress, reduceMotion: reduceMotion)
  }
}

struct TetherProgressContent: View {
  let progress: SyncProgress?
  let reduceMotion: Bool
  @Environment(\.colorScheme) private var scheme

  var body: some View {
    let theme = GardenTheme(scheme: scheme)
    Group {
      if let progress {
        VStack(spacing: 8) {
          HStack(spacing: 12) {
            Text(progress.label).foregroundStyle(theme.muted).lineLimit(1)
            Spacer(minLength: 8)
            TetherFlipReadout(
              value: readout(progress), reservation: reservation(progress),
              reduceMotion: reduceMotion
            )
            .foregroundStyle(theme.ink)
          }
          .font(GardenTheme.meta)
          .frame(height: 16)
          GeometryReader { geometry in
            ZStack(alignment: .leading) {
              Rectangle().fill(theme.line)
              if let fraction = progress.fraction {
                Rectangle().fill(theme.muted)
                  .frame(width: geometry.size.width * min(1, max(0, fraction)))
              } else {
                Path { path in
                  path.move(to: CGPoint(x: 0, y: 1))
                  path.addLine(to: CGPoint(x: geometry.size.width, y: 1))
                }
                .stroke(theme.muted, style: StrokeStyle(lineWidth: 2, dash: [2, 5]))
              }
            }
          }.frame(height: 2)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(progress.label)
        .accessibilityValue(accessibleValue(progress))
      } else {
        Color.clear.accessibilityHidden(true)
      }
    }
    .frame(height: 32)
    .transaction { transaction in
      if reduceMotion { transaction.animation = nil }
    }
  }

  private func readout(_ progress: SyncProgress) -> String {
    guard let total = progress.total else { return "–" }
    let count = "\(progress.completed) / \(total)"
    guard let fraction = progress.fraction else { return count }
    return "\(count) · \(Int(fraction * 100))%"
  }

  private func reservation(_ progress: SyncProgress) -> String {
    guard let total = progress.total else { return "000 / 000 · 100%" }
    let widest = String(max(total, progress.completed))
    return "\(widest) / \(widest) · 100%"
  }

  private func accessibleValue(_ progress: SyncProgress) -> String {
    guard let total = progress.total else {
      switch progress.phase {
      case .complete, .cancelled, .failed: return ""
      default: return "in progress"
      }
    }
    let count = "\(progress.completed) of \(total)"
    guard let fraction = progress.fraction else { return count }
    return "\(count), \(Int(fraction * 100)) percent"
  }
}

private struct TetherFlipReadout: View {
  let value: String
  let reservation: String
  let reduceMotion: Bool

  var body: some View {
    Text(reservation).hidden()
      .overlay(alignment: .trailing) {
        ZStack(alignment: .trailing) {
          Text(value)
            .id(value)
            .transition(
              reduceMotion
                ? .identity
                : .asymmetric(
                  insertion: .modifier(
                    active: ReadoutFlip(angle: -90, opacity: 0),
                    identity: ReadoutFlip(angle: 0, opacity: 1)),
                  removal: .modifier(
                    active: ReadoutFlip(angle: 90, opacity: 0),
                    identity: ReadoutFlip(angle: 0, opacity: 1))))
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.12), value: value)
      }
      .monospacedDigit()
      .fixedSize()
      .accessibilityHidden(true)
  }
}

private struct ReadoutFlip: ViewModifier {
  let angle: Double
  let opacity: Double

  func body(content: Content) -> some View {
    content
      .rotation3DEffect(.degrees(angle), axis: (x: 1, y: 0, z: 0), perspective: 0.25)
      .opacity(opacity)
  }
}
