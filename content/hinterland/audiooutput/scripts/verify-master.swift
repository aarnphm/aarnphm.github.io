// Failure coverage: unrelated media keys, key-up, repeat mute, bounds, mute restore,
// and master gain composition. Hardware capture is covered separately by --probe-device.
import Foundation

@main struct VerifyMaster {
    static func main() {
        assert(VolumeKey.decode(subtype: 8, data: (0 << 16) | 0x0a00) == .up)
        assert(VolumeKey.decode(subtype: 8, data: (1 << 16) | 0x0a01) == .down)
        assert(VolumeKey.decode(subtype: 8, data: (7 << 16) | 0x0a00) == .mute)
        assert(VolumeKey.decode(subtype: 8, data: (7 << 16) | 0x0a01) == nil)
        assert(VolumeKey.decode(subtype: 8, data: (0 << 16) | 0x0b00) == nil)
        assert(VolumeKey.decode(subtype: 8, data: (16 << 16) | 0x0a00) == nil)
        assert(VolumeKey.decode(subtype: 1, data: 0x0a00) == nil)
        var level = MasterLevel(volume: 0.5)
        assert(level.gain * 2 == 1)
        level.adjust(.mute)
        assert(level.gain == 0)
        level.adjust(.mute)
        assert(level.gain == 0.5)
        level.adjust(.down)
        assert(level.gain == 0.4375)
        level.adjust(.up, fine: true)
        assert(level.gain == 0.453125)
        for _ in 0..<100 { level.adjust(.down) }
        assert(level.gain == 0)
        for _ in 0..<100 { level.adjust(.up) }
        assert(level.gain == 1)
        let data = try! JSONEncoder().encode(level)
        assert(try! JSONDecoder().decode(MasterLevel.self, from: data).gain == 1)
        print("PASS: media key decoding, repeat handling, mute restore, bounds, fine steps, gain composition, persistence")
    }
}
