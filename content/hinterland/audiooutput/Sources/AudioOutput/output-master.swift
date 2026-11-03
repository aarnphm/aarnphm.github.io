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
    var onChange: (() -> Void)?
    private let keys = VolumeKeys()
    private var device: OutputDevice?
    private var paused = false
    private var route: AudioRoute?
    private var failedSignature: String?
    private let settingsKey = "audiooutput.master-levels.v1"
    private let enabledKey = "audiooutput.volume-keys.enabled"
    private var levels: [String: MasterLevel]

    init() {
        enabled = UserDefaults.standard.object(forKey: enabledKey) as? Bool ?? true
        if let data = UserDefaults.standard.data(forKey: settingsKey),
           let saved = try? JSONDecoder().decode([String: MasterLevel].self, from: data) {
            levels = saved
        } else { levels = [:] }
        keys.onKey = { [weak self] key, fine in
            guard let self, self.canControl else { return }
            self.level.adjust(key, fine: fine)
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

    func configure(device next: OutputDevice?, paused: Bool) {
        if device != next {
            stopRoute()
            failedSignature = nil
            device = next
            level = next.flatMap { levels[$0.uid] } ?? MasterLevel()
        }
        self.paused = paused
        eligible = next.map { !Hardware.hasOutputVolume($0.id) && Hardware.hasStereoOutput($0.id) } ?? false
        if enabled { keys.start() }
        keyAccess = keys.available
        if !canControl { stopRoute() }
    }

    func factor(for output: OutputDevice) -> Double {
        canControl && output.uid == device?.uid ? level.gain : 1
    }

    func apply(excluding processes: [AudioObjectID]) {
        guard canControl, let device else { stopRoute(); return }
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
}
