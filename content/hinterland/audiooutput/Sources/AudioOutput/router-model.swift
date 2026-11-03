import AppKit
import CoreAudio
import Combine
import ServiceManagement

struct Assignment: Codable, Identifiable {
    var id: String
    var name: String
    var deviceUID: String
    var deviceName: String
    var volume: Double? = nil
    var gain: Double { min(3, max(0, volume ?? 1)) }
}

@available(macOS 14.2, *)
@MainActor
final class RouterModel: ObservableObject {
    let master = OutputMaster()
    @Published var siteRules = SiteRouting.load().sites
    @Published var siteStatuses: [String: String] = [:]
    @Published var defaultOutputName = ""
    @Published var outputs: [OutputDevice] = []
    @Published var applications: [AudioApplication] = []
    @Published var assignments: [Assignment] = []
    @Published var statuses: [String: String] = [:]
    @Published var issue: String?
    @Published var paused = false
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled
    @Published var activeCount = 0
    private var routes: [String: AudioRoute] = [:]
    private var failures: [String: String] = [:]
    private var defaultOutputID: AudioObjectID = 0
    private var sleeping = false
    private var timer: Timer?
    private var observers: [NSObjectProtocol] = []
    private let key = "audiooutput.assignments.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: key) {
            do { assignments = try JSONDecoder().decode([Assignment].self, from: data) }
            catch { issue = "Saved assignments could not be read: \(error.localizedDescription)" }
        }
        master.onChange = { [weak self] in self?.refresh() }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
        for event in [NSWorkspace.willSleepNotification, NSWorkspace.didWakeNotification] {
            observers.append(NSWorkspace.shared.notificationCenter.addObserver(forName: event, object: nil, queue: .main) { [weak self] note in
                Task { @MainActor in
                    guard let self else { return }
                    if note.name == NSWorkspace.willSleepNotification { self.sleeping = true; self.stopAll(); self.publishSites() }
                    else { self.sleeping = false; self.failures.removeAll(); self.refresh() }
                }
            })
        }
    }

    func refresh() {
        guard !sleeping else { return }
        do {
            outputs = try Hardware.outputs()
            let defaultID = try Hardware.scalar(Hardware.system, kAudioHardwarePropertyDefaultOutputDevice, initial: AudioObjectID(0))
            defaultOutputID = defaultID
            master.configure(device: outputs.first { $0.id == defaultID }, paused: paused)
            defaultOutputName = outputs.first { $0.id == defaultID }?.name ?? "Unavailable"
            for site in SiteRouting.load().sites where !siteRules.contains(where: { $0.id == site.id }) {
                siteRules.append(site)
            }
            applications = try Hardware.applications()
            for app in applications {
                guard let host = app.siteHost, SiteRouting.validHost(host) else { continue }
                if !siteRules.contains(where: { $0.id == host }) {
                    siteRules.append(SiteRule(id: host, name: app.name, outputUID: "", outputName: "Normal output"))
                }
                if let old = assignments.first(where: { $0.id == app.id }),
                   let index = siteRules.firstIndex(where: { $0.id == host }) {
                    siteRules[index].outputUID = old.deviceUID
                    siteRules[index].outputName = old.deviceName
                    siteRules[index].volume = old.volume
                    assignments.removeAll { $0.id == app.id }
                    persist()
                }
            }
            publishSites()
            reconcile()
        } catch { issue = error.localizedDescription; stopAll() }
    }

    func assignSite(_ site: SiteRule, uid: String) {
        guard let index = siteRules.firstIndex(where: { $0.id == site.id }) else { return }
        siteRules[index].outputUID = uid
        siteRules[index].outputName = outputs.first { $0.uid == uid }?.name ?? "Normal output"
        publishSites()
    }

    private func publishSites() {
        do {
            try SiteRouting.save(SiteConfiguration(enabled: !paused && !sleeping, sites: siteRules))
            siteStatuses = Dictionary(uniqueKeysWithValues: siteRules.map { site in
                let status = SiteRouting.status(site.id)
                let detail: String
                if paused { detail = "Paused" }
                else if site.outputUID.isEmpty && site.gain == 1 { detail = "Normal output" }
                else if let status, status.outputUID == site.outputUID { detail = status.detail }
                else { detail = "Waiting for browser extension" }
                return (site.id, detail)
            })
        } catch { issue = error.localizedDescription }
    }

    func setVolume(_ app: AudioApplication, percent: Double) {
        let gain = min(300, max(0, percent)) / 100
        if let index = assignments.firstIndex(where: { $0.id == app.id }) {
            assignments[index].volume = gain
            if gain == 1 && assignments[index].deviceUID.isEmpty {
                assignments.remove(at: index)
                routes.removeValue(forKey: app.id)?.stop()
                failures.removeValue(forKey: app.id)
            }
        } else if gain != 1 {
            assignments.append(Assignment(id: app.id, name: app.name, deviceUID: "", deviceName: "Normal output", volume: gain))
        }
        persist()
        reconcile()
    }

    func setSiteVolume(_ site: SiteRule, percent: Double) {
        guard let index = siteRules.firstIndex(where: { $0.id == site.id }) else { return }
        siteRules[index].volume = min(300, max(0, percent)) / 100
        publishSites()
    }

    func assign(_ app: AudioApplication, uid: String) {
        let gain = assignments.first { $0.id == app.id }?.gain ?? 1
        routes.removeValue(forKey: app.id)?.stop()
        failures.removeValue(forKey: app.id)
        assignments.removeAll { $0.id == app.id }
        if let device = outputs.first(where: { $0.uid == uid }) {
            assignments.append(Assignment(id: app.id, name: app.name, deviceUID: uid, deviceName: device.name, volume: gain))
        } else if gain != 1 {
            assignments.append(Assignment(id: app.id, name: app.name, deviceUID: "", deviceName: "Normal output", volume: gain))
        }
        persist()
        reconcile()
    }

    func forget(_ id: String) {
        routes.removeValue(forKey: id)?.stop()
        failures.removeValue(forKey: id)
        assignments.removeAll { $0.id == id }
        persist()
        reconcile()
    }

    func togglePause() { paused.toggle(); refresh() }
    func retry() { failures.removeAll(); master.retry(); issue = nil; refresh() }

    func setLogin(_ enabled: Bool) {
        do {
            if enabled { try SMAppService.mainApp.register() }
            else { try SMAppService.mainApp.unregister() }
            launchAtLogin = SMAppService.mainApp.status == .enabled
            if SMAppService.mainApp.status == .requiresApproval { SMAppService.openSystemSettingsLoginItems() }
        } catch { issue = error.localizedDescription }
    }

    private func persist() {
        do { UserDefaults.standard.set(try JSONEncoder().encode(assignments), forKey: key) }
        catch { issue = "Could not save assignments: \(error.localizedDescription)" }
    }

    private func reconcile() {
        var nextStatuses: [String: String] = [:]
        for assignment in assignments {
            let app = applications.first { $0.id == assignment.id }
            let device = outputs.first { assignment.deviceUID.isEmpty ? $0.id == defaultOutputID : $0.uid == assignment.deviceUID }
            guard !paused, let app, !app.processes.isEmpty, let device else {
                routes.removeValue(forKey: assignment.id)?.stop()
                failures.removeValue(forKey: assignment.id)
                if let owner = app?.sharedAudioOwner {
                    nextStatuses[assignment.id] = "Shares \(owner) audio"
                    continue
                }
                nextStatuses[assignment.id] = paused ? "Paused · normal output" : device == nil
                    ? "Disconnected · normal output" : "Waiting for application audio"
                continue
            }
            if let owner = app.sharedAudioOwner {
                routes.removeValue(forKey: app.id)?.stop()
                nextStatuses[app.id] = "Shares \(owner) audio"
                continue
            }
            let signature = "\(app.processes)-\(device.id)-\(device.sampleRate)"
            if let route = routes[app.id], route.source != .processes(app.processes) || route.device != device {
                routes.removeValue(forKey: app.id)?.stop()
                failures.removeValue(forKey: app.id)
            }
            if routes[app.id] == nil && failures[app.id] != signature {
                do { routes[app.id] = try AudioRoute(source: .processes(app.processes), device: device, gain: Float(assignment.gain * master.factor(for: device))) }
                catch {
                    failures[app.id] = signature
                    statuses[app.id] = error.localizedDescription
                }
            }
            if let route = routes[app.id] {
                route.setGain(Float(assignment.gain * master.factor(for: device)))
                nextStatuses[app.id] = route.nonzeroFrames > 0 ? "Routing to \(device.name)" : "Listening · waiting for audio"
            } else { nextStatuses[app.id] = statuses[app.id] ?? "Route unavailable · retry" }
        }
        let excluded = routes.values.flatMap { route -> [AudioObjectID] in
            if case .processes(let processes) = route.source { return processes }
            return []
        }
        master.apply(excluding: excluded)
        statuses = nextStatuses
        activeCount = routes.count + (master.active ? 1 : 0) + (paused ? 0 : siteRules.filter { site in
            guard !site.outputUID.isEmpty || site.gain != 1, let status = SiteRouting.status(site.id) else { return false }
            return status.state == "routing" && status.outputUID == site.outputUID
        }.count)
    }

    func stopAll() {
        master.stopRoute()
        for route in routes.values { route.stop() }
        routes.removeAll()
        activeCount = 0
    }

    func shutdown() {
        master.shutdown()
        timer?.invalidate()
        for observer in observers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        observers.removeAll()
        stopAll()
        try? SiteRouting.save(SiteConfiguration(enabled: false, sites: siteRules))
    }
}
