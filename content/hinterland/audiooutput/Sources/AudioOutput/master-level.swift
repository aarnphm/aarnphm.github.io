import Foundation

enum VolumeKey: Equatable {
    case up, down, mute

    static func decode(subtype: Int16, data: Int) -> Self? {
        guard subtype == 8, (data >> 8) & 0xff == 0x0a else { return nil }
        switch (data >> 16) & 0xffff {
        case 0: return .up
        case 1: return .down
        case 7: return data & 1 == 0 ? .mute : nil
        default: return nil
        }
    }
}

struct MasterLevel: Codable {
    var volume: Double = 1
    var muted = false
    var gain: Double { muted ? 0 : min(1, max(0, volume)) }

    mutating func adjust(_ key: VolumeKey, fine: Bool = false) {
        let step = fine ? 1.0 / 64 : 1.0 / 16
        switch key {
        case .up:
            volume = min(1, gain + step)
            muted = false
        case .down:
            volume = max(0, gain - step)
            muted = false
        case .mute: muted.toggle()
        }
    }
}
