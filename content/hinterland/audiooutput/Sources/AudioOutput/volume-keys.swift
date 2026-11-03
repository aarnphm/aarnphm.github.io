import AppKit
import ApplicationServices

// The tap runs on the main run loop. Its callback only decodes media keys and
// queues the adjustment; route creation never blocks the event tap.
@MainActor
final class VolumeKeys {
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    var handlesKeys = false
    var onKey: ((VolumeKey, Bool) -> Void)?
    var available: Bool { tap != nil }

    func start() {
        guard tap == nil, AXIsProcessTrusted() else { return }
        let mask = CGEventMask(1) << NSEvent.EventType.systemDefined.rawValue
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap,
            options: .defaultTap, eventsOfInterest: mask, callback: { _, type, event, context in
                guard let context else { return Unmanaged.passUnretained(event) }
                return MainActor.assumeIsolated {
                    let owner = Unmanaged<VolumeKeys>.fromOpaque(context).takeUnretainedValue()
                    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                        if let tap = owner.tap { CGEvent.tapEnable(tap: tap, enable: true) }
                        return Unmanaged.passUnretained(event)
                    }
                    guard owner.handlesKeys, let keyEvent = NSEvent(cgEvent: event),
                          keyEvent.subtype.rawValue == 8 else { return Unmanaged.passUnretained(event) }
                    let code = (keyEvent.data1 >> 16) & 0xffff
                    guard [0, 1, 7].contains(code) else { return Unmanaged.passUnretained(event) }
                    if let key = VolumeKey.decode(subtype: keyEvent.subtype.rawValue, data: keyEvent.data1) {
                        let fine = keyEvent.modifierFlags.contains([.shift, .option])
                        DispatchQueue.main.async { owner.onKey?(key, fine) }
                    }
                    return nil
                }
            }, userInfo: Unmanaged.passUnretained(self).toOpaque()) else { return }
        self.tap = tap
        source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        if let source { CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes) }
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    func requestAccess() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
        start()
    }

    func stop() {
        handlesKeys = false
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let tap { CGEvent.tapEnable(tap: tap, enable: false); CFMachPortInvalidate(tap) }
        source = nil
        tap = nil
    }
}
