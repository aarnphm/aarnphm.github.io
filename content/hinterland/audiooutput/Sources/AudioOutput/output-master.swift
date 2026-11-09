import AppKit
import CoreAudio
import Combine

@available(macOS 14.2, *)
@MainActor
final class OutputMaster: ObservableObject {
    @Published private(set) var level = MasterLevel()
    @Published private(set) var eligible = false
    @Published private(set) var keyAccess = false
    @Published private(set) var issue: String?
    @Published private(set) var enabled: Bool
    @Published private(set) var targetUID: String
    @Published private(set) var targetName: String
    var onChange: (() -> Void)?
    private let keys = VolumeKeys()
    private var device: OutputDevice?
    private var paused = false
    private var route: AudioRoute?
    private var hardwareVolume: HardwareVolume?
    private var failedSignature: String?
    private let settingsKey = "audiooutput.master-levels.v1"
    private let enabledKey = "audiooutput.volume-keys.enabled"
    private let targetKey = "audiooutput.volume-keys.target"
    private let targetNameKey = "audiooutput.volume-keys.target-name"
    private var levels: [String: MasterLevel]

    init() {
        enabled = UserDefaults.standard.object(forKey: enabledKey) as? Bool ?? true
        targetUID = UserDefaults.standard.string(forKey: targetKey) ?? ""
        targetName = UserDefaults.standard.string(forKey: targetNameKey) ?? "Saved output"
        if let data = UserDefaults.standard.data(forKey: settingsKey),
           let saved = try? JSONDecoder().decode([String: MasterLevel].self, from: data) {
            levels = saved
        } else { levels = [:] }
        keys.onKey = { [weak self] key, fine in
            guard let self, self.canControl else { return }
            var next = self.level
            do {
                if let hardware = self.hardwareVolume {
                    let current = try hardware.level()
                    if !next.muted || current > 0 {
                        next.volume = current
                        next.muted = false
                    }
                }
                next.adjust(key, fine: fine)
                try self.hardwareVolume?.setLevel(next.gain)
                self.level = next
                self.issue = nil
            } catch {
                self.issue = error.localizedDescription
                return
            }
            if let device = self.device {
                self.levels[device.uid] = self.level
                if let data = try? JSONEncoder().encode(self.levels) {
                    UserDefaults.standard.set(data, forKey: self.settingsKey)
                }
            }
            self.route?.setGain(Float(self.level.gain))
            self.onChange?()
        }
    }

    private var canControl: Bool { enabled && eligible && keyAccess && !paused }
    var active: Bool { route != nil }

    func configure(outputs: [OutputDevice], defaultID: AudioObjectID, paused: Bool) {
        let next = outputs.first { targetUID.isEmpty ? $0.id == defaultID : $0.uid == targetUID }
        if device != next {
            stopRoute()
            failedSignature = nil
            issue = nil
            device = next
            hardwareVolume = next.flatMap { HardwareVolume(device: $0.id) }
            level = next.flatMap { levels[$0.uid] } ?? MasterLevel()
        }
        if let hardwareVolume, let current = try? hardwareVolume.level(), !level.muted || current > 0 {
            level = MasterLevel(volume: current)
        }
        self.paused = paused
        eligible = next.map { hardwareVolume != nil ? !targetUID.isEmpty : Hardware.hasStereoOutput($0.id) } ?? false
        if enabled { keys.start() }
        keyAccess = keys.available
        if !canControl { stopRoute() }
    }

    func selectTarget(_ output: OutputDevice?) {
        targetUID = output?.uid ?? ""
        targetName = output?.name ?? "System output"
        UserDefaults.standard.set(targetUID, forKey: targetKey)
        UserDefaults.standard.set(targetName, forKey: targetNameKey)
        onChange?()
    }

    func factor(for output: OutputDevice) -> Double {
        canControl && hardwareVolume == nil && output.uid == device?.uid ? level.gain : 1
    }

    func apply(excluding processes: [AudioObjectID]) {
        defer { writeStatus() }
        guard canControl, let device else { stopRoute(); return }
        if hardwareVolume != nil {
            stopRoute()
            keys.handlesKeys = true
            return
        }
        let source = AudioRouteSource.device(excluding: Array(Set(processes)).sorted())
        if let route, route.source != source || route.device != device { stopRoute(); failedSignature = nil }
        let signature = "\(source)-\(device.id)-\(device.sampleRate)"
        if route == nil && failedSignature != signature {
            do {
                route = try AudioRoute(source: source, device: device, gain: Float(level.gain))
                issue = nil
            } catch {
                failedSignature = signature
                issue = error.localizedDescription
            }
        }
        route?.setGain(Float(level.gain))
        keys.handlesKeys = route != nil
    }

    func setEnabled(_ value: Bool) {
        enabled = value
        UserDefaults.standard.set(value, forKey: enabledKey)
        if !value { stopRoute(); keys.stop(); keyAccess = false; issue = nil }
        onChange?()
    }

    func requestAccess() { keys.requestAccess(); onChange?() }
    func retry() { failedSignature = nil; issue = nil }

    func stopRoute() {
        keys.handlesKeys = false
        route?.stop()
        route = nil
    }

    func shutdown() { stopRoute(); keys.stop() }

    private func writeStatus() {
        struct Status: Encodable {
            let pid: Int32
            let updatedAt: Date
            let target: String?
            let enabled: Bool
            let eligible: Bool
            let trusted: Bool
            let keyTap: Bool
            let handlesKeys: Bool
            let gain: Double
            let volumeControl: String
            let frames: UInt64
            let nonzeroFrames: UInt64
            let issue: String?
        }
        let status = Status(pid: ProcessInfo.processInfo.processIdentifier, updatedAt: Date(),
            target: device?.uid, enabled: enabled, eligible: eligible, trusted: keys.trusted,
            keyTap: keys.available, handlesKeys: keys.handlesKeys, gain: level.gain,
            volumeControl: hardwareVolume == nil ? "software" : "hardware",
            frames: route?.frames ?? 0, nonzeroFrames: route?.nonzeroFrames ?? 0,
            issue: issue ?? keys.issue)
        if let data = try? JSONEncoder().encode(status) {
            try? data.write(to: SiteRouting.directory.appendingPathComponent("master-status.json"), options: .atomic)
        }
    }
}
