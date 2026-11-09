import Foundation
import CoreAudio

@main
struct VerifyHardwareVolume {
    static func main() throws {
        guard CommandLine.arguments.count == 2,
              let device = try Hardware.outputs().first(where: { $0.uid == CommandLine.arguments[1] }),
              let control = HardwareVolume(device: device.id) else {
            throw AudioFailure(message: "Supply a connected output UID with writable volume.")
        }
        let defaultBefore = try Hardware.scalar(Hardware.system, kAudioHardwarePropertyDefaultOutputDevice, initial: AudioObjectID(0))
        let tapsBefore = try Hardware.objects(Hardware.system, kAudioHardwarePropertyTapList)
        let original = try control.level()
        let lowered = max(0, original - 1.0 / 16)
        do {
            try control.setLevel(lowered)
            Thread.sleep(forTimeInterval: 0.3)
            let actual = try control.level()
            guard abs(actual - lowered) < 0.005 else { throw AudioFailure(message: "Volume readback differs: \(actual), expected \(lowered)") }
            try control.setLevel(original)
            Thread.sleep(forTimeInterval: 0.3)
            guard abs(try control.level() - original) < 0.005 else { throw AudioFailure(message: "Restore failed") }
            guard try Hardware.scalar(Hardware.system, kAudioHardwarePropertyDefaultOutputDevice, initial: AudioObjectID(0)) == defaultBefore,
                  try Hardware.objects(Hardware.system, kAudioHardwarePropertyTapList) == tapsBefore else {
                throw AudioFailure(message: "Default output or tap list changed")
            }
            print("PASS \(device.name): \(original) -> \(actual) -> \(original); default and audio taps unchanged")
        } catch {
            try control.setLevel(original)
            throw error
        }
    }
}
