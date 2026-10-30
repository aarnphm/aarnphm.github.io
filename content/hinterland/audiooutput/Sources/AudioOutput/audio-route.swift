import CoreAudio
import AudioRender
import Foundation

@available(macOS 14.2, *)
final class AudioRoute {
    let processes: [AudioObjectID]
    let device: OutputDevice
    private var tap: AudioObjectID = 0
    private var aggregate: AudioObjectID = 0
    private var callback: AudioDeviceIOProcID?
    private var renderState: OpaquePointer?
    var resourceIDs: [AudioObjectID] { [tap, aggregate].filter { $0 != 0 } }
    var frames: UInt64 { renderState.map(ao_render_frames) ?? 0 }
    var nonzeroFrames: UInt64 { renderState.map(ao_render_nonzero_frames) ?? 0 }

    init(processes: [AudioObjectID], device: OutputDevice, gain: Float = 1, progress: (String) -> Void = { _ in }) throws {
        self.processes = processes
        self.device = device
        do { try start(gain: gain, progress: progress) } catch { stop(); throw error }
    }

    private func start(gain: Float, progress: (String) -> Void) throws {
        progress("Creating process tap")
        let description = CATapDescription(stereoMixdownOfProcesses: processes)
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
