import AppKit
import CoreAudio
import Darwin

struct AudioFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

func checkAudio(_ status: OSStatus, _ operation: String) throws {
    guard status == noErr else { throw AudioFailure(message: "\(operation) failed (Core Audio \(status)).") }
}

struct OutputDevice: Identifiable, Equatable {
    let id: AudioObjectID
    let uid: String
    let name: String
    let sampleRate: Double
}

struct AudioApplication: Identifiable, Equatable {
    let id: String
    let name: String
    let processes: [AudioObjectID]
    let icon: NSImage?
    let sharedAudioOwner: String?
    let siteHost: String?
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id && lhs.name == rhs.name && lhs.processes == rhs.processes && lhs.sharedAudioOwner == rhs.sharedAudioOwner
    }
}

enum Hardware {
    static let system = AudioObjectID(kAudioObjectSystemObject)

    static func address(_ selector: AudioObjectPropertySelector,
                        _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }

    static func scalar<T: BitwiseCopyable>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                          initial: T, scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) throws -> T {
        var value = initial
        var size = UInt32(MemoryLayout<T>.size)
        var property = address(selector, scope)
        try checkAudio(AudioObjectGetPropertyData(object, &property, 0, nil, &size, &value), "Read audio property")
        return value
    }

    static func string(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> String {
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var property = address(selector)
        try checkAudio(AudioObjectGetPropertyData(object, &property, 0, nil, &size, &value), "Read audio name")
        guard let value else { throw AudioFailure(message: "Audio property returned no string.") }
        return value.takeRetainedValue() as String
    }

    static func objects(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) throws -> [AudioObjectID] {
        var property = address(selector, scope)
        var size: UInt32 = 0
        try checkAudio(AudioObjectGetPropertyDataSize(object, &property, 0, nil, &size), "Read audio list size")
        var values = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard size > 0 else { return [] }
        try values.withUnsafeMutableBytes { buffer in
            try checkAudio(AudioObjectGetPropertyData(object, &property, 0, nil, &size, buffer.baseAddress!), "Read audio list")
        }
        return Array(values.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
    }

    static func channels(_ device: AudioObjectID, scope: AudioObjectPropertyScope) throws -> UInt32 {
        var property = address(kAudioDevicePropertyStreamConfiguration, scope)
        var size: UInt32 = 0
        try checkAudio(AudioObjectGetPropertyDataSize(device, &property, 0, nil, &size), "Read channel layout")
        guard size >= MemoryLayout<UInt32>.size else { return 0 }
        let storage = UnsafeMutableRawPointer.allocate(byteCount: max(Int(size), MemoryLayout<AudioBufferList>.size),
                                                       alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { storage.deallocate() }
        try checkAudio(AudioObjectGetPropertyData(device, &property, 0, nil, &size, storage), "Read channels")
        return UnsafeMutableAudioBufferListPointer(storage.assumingMemoryBound(to: AudioBufferList.self))
            .reduce(0) { $0 + $1.mNumberChannels }
    }

    static func outputs() throws -> [OutputDevice] {
        try objects(system, kAudioHardwarePropertyDevices).compactMap { device in
            guard (try? channels(device, scope: kAudioDevicePropertyScopeOutput)) ?? 0 > 0,
                  let uid = try? string(device, kAudioDevicePropertyDeviceUID),
                  !uid.hasPrefix("xyz.aarnphm.audiooutput.route."),
                  let name = try? string(device, kAudioObjectPropertyName),
                  let rate = try? scalar(device, kAudioDevicePropertyNominalSampleRate, initial: Double(0)) else { return nil }
            return OutputDevice(id: device, uid: uid, name: name, sampleRate: rate)
        }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    static func applications() throws -> [AudioApplication] {
        var siteHosts: [String: String] = [:]
        var sharedOwners: [String: String] = [:]
        var grouped: [String: (String, [AudioObjectID], NSImage?)] = [:]
        for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
            guard let bundle = app.bundleIdentifier, bundle != "xyz.aarnphm.audiooutput" else { continue }
            if let url = app.bundleURL, let metadata = Bundle(url: url),
               let ownerID = metadata.object(forInfoDictionaryKey: "CrBundleIdentifier") as? String {
                if let address = metadata.object(forInfoDictionaryKey: "CrAppModeShortcutURL") as? String,
                   let host = URL(string: address)?.host { siteHosts[bundle] = host }
                sharedOwners[bundle] = NSWorkspace.shared.runningApplications
                    .first { $0.bundleIdentifier == ownerID }?.localizedName ?? ownerID
            }
            grouped[bundle] = (app.localizedName ?? bundle, [], app.icon)
        }
        for process in try objects(system, kAudioHardwarePropertyProcessObjectList) {
            guard let pid = try? scalar(process, kAudioProcessPropertyPID, initial: pid_t(0)),
                  pid != ProcessInfo.processInfo.processIdentifier else { continue }
            let running = NSRunningApplication(processIdentifier: pid)
            let owner = applicationBundle(pid: pid)
            let bundle = owner?.bundleIdentifier ?? running?.bundleIdentifier
                ?? (try? string(process, kAudioProcessPropertyBundleID)) ?? ""
            guard !bundle.isEmpty, bundle != "xyz.aarnphm.audiooutput" else { continue }
            let name = owner?.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
                ?? owner?.object(forInfoDictionaryKey: "CFBundleName") as? String
                ?? running?.localizedName ?? bundle
            let icon = owner.map { NSWorkspace.shared.icon(forFile: $0.bundlePath) } ?? running?.icon
            var entry = grouped[bundle] ?? (name, [], icon)
            entry.1.append(process)
            grouped[bundle] = entry
        }
        return grouped.map { AudioApplication(id: $0.key, name: $0.value.0,
                                              processes: $0.value.1.sorted(), icon: $0.value.2,
                                              sharedAudioOwner: sharedOwners[$0.key], siteHost: siteHosts[$0.key]) }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
    private static func applicationBundle(pid: pid_t) -> Bundle? {
        var path = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        let length = path.withUnsafeMutableBytes { proc_pidpath(pid, $0.baseAddress, UInt32($0.count)) }
        guard length > 0 else { return nil }
        let executable = String(cString: path)
        var url = URL(fileURLWithPath: "/")
        for component in executable.split(separator: "/") {
            url.appendPathComponent(String(component))
            if component.hasSuffix(".app") { return Bundle(url: url) }
        }
        return nil
    }

}
