import CoreAudio

// Prefer the device's master control; otherwise keep its channel balance.
final class HardwareVolume {
    private let device: AudioObjectID
    private let elements: [UInt32]
    private var lastAudible: [Float] = []

    init?(device: AudioObjectID) {
        self.device = device
        let channels = (try? Hardware.channels(device, scope: kAudioDevicePropertyScopeOutput)) ?? 0
        let candidates: [UInt32] = channels > 0 ? Array(0...channels) : [0]
        let writable = candidates.filter { element in
            var property = Hardware.address(kAudioDevicePropertyVolumeScalar, kAudioDevicePropertyScopeOutput)
            property.mElement = element
            var settable: DarwinBoolean = false
            return AudioObjectHasProperty(device, &property)
                && AudioObjectIsPropertySettable(device, &property, &settable) == noErr && settable.boolValue
        }
        guard !writable.isEmpty else { return nil }
        elements = writable.contains(0) ? [0] : writable
    }

    private func values() throws -> [Float] {
        try elements.map { element in
            var property = Hardware.address(kAudioDevicePropertyVolumeScalar, kAudioDevicePropertyScopeOutput)
            property.mElement = element
            var value: Float = 0
            var size = UInt32(MemoryLayout<Float>.size)
            try checkAudio(AudioObjectGetPropertyData(device, &property, 0, nil, &size, &value), "Read device volume")
            return value
        }
    }

    func level() throws -> Double { Double(try values().max() ?? 0) }

    func setLevel(_ level: Double) throws {
        let previous = try values()
        if (previous.max() ?? 0) > 0 { lastAudible = previous }
        let reference = lastAudible.isEmpty ? Array(repeating: Float(1), count: elements.count) : lastAudible
        let peak = reference.max() ?? 1
        let next = reference.map { Float(min(1, max(0, level))) * $0 / peak }
        do { try write(next) }
        catch { try? write(previous); throw error }
    }

    private func write(_ values: [Float]) throws {
        for (element, value) in zip(elements, values) {
            var property = Hardware.address(kAudioDevicePropertyVolumeScalar, kAudioDevicePropertyScopeOutput)
            property.mElement = element
            var scalar = value
            try checkAudio(AudioObjectSetPropertyData(device, &property, 0, nil,
                UInt32(MemoryLayout<Float>.size), &scalar), "Set device volume")
        }
    }
}
