import CoreAudio
import AudioRender
import Foundation

enum AudioRouteSource: Equatable {
    case processes([AudioObjectID])
    case device(excluding: [AudioObjectID])
}

@available(macOS 14.2, *)
final class AudioRoute {
    let source: AudioRouteSource
    let device: OutputDevice
    private var tap: AudioObjectID = 0
    private var aggregate: AudioObjectID = 0
    private var callback: AudioDeviceIOProcID?
    private var renderState: OpaquePointer?
    var resourceIDs: [AudioObjectID] { [tap, aggregate].filter { $0 != 0 } }
    var frames: UInt64 { renderState.map(ao_render_frames) ?? 0 }
    var nonzeroFrames: UInt64 { renderState.map(ao_render_nonzero_frames) ?? 0 }

    init(source: AudioRouteSource, device: OutputDevice, gain: Float = 1, progress: (String) -> Void = { _ in }) throws {
        self.source = source
        self.device = device
        do { try start(gain: gain, progress: progress) } catch { stop(); throw error }
    }

    private func start(gain: Float, progress: (String) -> Void) throws {
        progress("Creating process tap")
        let description: CATapDescription
        switch source {
        case .processes(let processes):
            description = CATapDescription(stereoMixdownOfProcesses: processes)
        case .device(let excluded):
            let streams = try Hardware.objects(device.id, kAudioDevicePropertyStreams, scope: kAudioDevicePropertyScopeOutput)
            guard streams.count == 1, let stream = streams.first else {
                throw AudioFailure(message: "Volume keys require a single stereo output stream.")
            }
            let format = try Hardware.scalar(stream, kAudioStreamPropertyVirtualFormat, initial: AudioStreamBasicDescription())
            guard format.mChannelsPerFrame == 2 else {
                throw AudioFailure(message: "Volume keys require a stereo output.")
            }
            description = CATapDescription(excludingProcesses: excluded, deviceUID: device.uid, stream: 0)
        }
        description.name = "Audio Output route"
        description.uuid = UUID()
        description.isPrivate = true
        description.muteBehavior = .mutedWhenTapped
        try checkAudio(AudioHardwareCreateProcessTap(description, &tap), "Create app audio tap")
        progress("Creating private aggregate")
        let uid = "xyz.aarnphm.audiooutput.route.\(UUID().uuidString)"
        let configuration: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Audio Output route",
            kAudioAggregateDeviceUIDKey: uid,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceMainSubDeviceKey: device.uid,
            kAudioAggregateDeviceSubDeviceListKey: [[
                kAudioSubDeviceUIDKey: device.uid,
                kAudioSubDeviceInputChannelsKey: 0,
            ]],
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapUIDKey: description.uuid.uuidString,
                kAudioSubTapDriftCompensationKey: true,
            ]],
            kAudioAggregateDeviceTapAutoStartKey: true,
        ]
        try checkAudio(AudioHardwareCreateAggregateDevice(configuration as CFDictionary, &aggregate), "Create output route")
        progress("Setting aggregate clock to \(device.sampleRate) Hz")
        var sampleRate = device.sampleRate
        var sampleRateProperty = Hardware.address(kAudioDevicePropertyNominalSampleRate)
        try checkAudio(AudioObjectSetPropertyData(aggregate, &sampleRateProperty, 0, nil,
            UInt32(MemoryLayout<Double>.size), &sampleRate), "Set route clock rate")
        progress("Validating stream formats")
        let inputChannels = try Hardware.channels(aggregate, scope: kAudioDevicePropertyScopeInput)
        let physicalInputs = try Hardware.channels(device.id, scope: kAudioDevicePropertyScopeInput)
        guard inputChannels == 2 || inputChannels == physicalInputs + 2 else {
            throw AudioFailure(message: "The output exposed an unsupported input layout. Normal playback is unchanged.")
        }
        let inputOffset = inputChannels - 2
        var rate: Double?
        for scope in [kAudioDevicePropertyScopeInput, kAudioDevicePropertyScopeOutput] {
            let streams = try Hardware.objects(aggregate, kAudioDevicePropertyStreams, scope: scope)
            guard !streams.isEmpty else { throw AudioFailure(message: "The route has no audio streams.") }
            for stream in streams {
                let format = try Hardware.scalar(stream, kAudioStreamPropertyVirtualFormat, initial: AudioStreamBasicDescription())
                guard format.mFormatID == kAudioFormatLinearPCM,
                      format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
                      format.mBitsPerChannel == 32,
                      format.mFormatFlags & kAudioFormatFlagIsBigEndian == 0,
                      format.mBytesPerFrame == 4 * (format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0 ? 1 : format.mChannelsPerFrame)
                else { throw AudioFailure(message: "This output does not expose 32-bit floating-point audio.") }
                if let rate, abs(rate - format.mSampleRate) > 1 {
                    throw AudioFailure(message: "The route's sample rates differ (\(rate) and \(format.mSampleRate) Hz). Reconnect the output and retry.")
                }
                rate = format.mSampleRate
            }
        }
        progress("Registering audio callback")
        guard let state = ao_render_create(inputOffset) else { throw AudioFailure(message: "Could not allocate audio state.") }
        ao_render_set_gain(state, gain)
        renderState = state
        try checkAudio(AudioDeviceCreateIOProcID(aggregate, ao_render, UnsafeMutableRawPointer(state), &callback), "Register audio renderer")
        guard let callback else { throw AudioFailure(message: "Audio callback was not created.") }
        try checkAudio(ao_select_tap_input(aggregate, callback, inputOffset), "Disable physical input streams")
        if case .device(let excluded) = source {
            let ownProcesses = try Hardware.objects(Hardware.system, kAudioHardwarePropertyProcessObjectList).filter {
                (try? Hardware.scalar($0, kAudioProcessPropertyPID, initial: pid_t(0))) == ProcessInfo.processInfo.processIdentifier
            }
            guard !ownProcesses.isEmpty else {
                throw AudioFailure(message: "Could not exclude the renderer from the output tap.")
            }
            description.processes = Array(Set(excluded + ownProcesses)).sorted()
            var updated = Unmanaged.passUnretained(description)
            var property = Hardware.address(kAudioTapPropertyDescription)
            try checkAudio(AudioObjectSetPropertyData(tap, &property, 0, nil,
                UInt32(MemoryLayout<Unmanaged<CATapDescription>>.size), &updated), "Exclude master renderer")
        }
        progress("Starting audio IO")
        try checkAudio(AudioDeviceStart(aggregate, callback), "Start audio routing (check System Audio Recording permission)")
    }

    func setGain(_ gain: Float) {
        if let renderState { ao_render_set_gain(renderState, gain) }
    }

    func stop() {
        if let callback {
            AudioDeviceStop(aggregate, callback)
            AudioDeviceDestroyIOProcID(aggregate, callback)
            self.callback = nil
        }
        if aggregate != 0 { AudioHardwareDestroyAggregateDevice(aggregate); aggregate = 0 }
        if tap != 0 { AudioHardwareDestroyProcessTap(tap); tap = 0 }
        if let renderState { ao_render_destroy(renderState); self.renderState = nil }
    }

    deinit { stop() }
}
