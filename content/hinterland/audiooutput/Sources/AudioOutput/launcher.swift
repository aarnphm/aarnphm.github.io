import AppKit
import CoreAudio

@main
struct Launcher {
    @MainActor static func main() {
        guard #available(macOS 14.2, *) else {
            fputs("Audio Output requires macOS 14.2 or later.\n", stderr)
            exit(1)
        }
        let arguments = CommandLine.arguments
        do {
            if arguments.contains("--native-host") { try NativeHost.run(); return }
            if arguments.contains("--inventory") {
                let defaultID = try Hardware.scalar(Hardware.system, kAudioHardwarePropertyDefaultOutputDevice, initial: AudioObjectID(0))
                print("DEFAULT_OUTPUT\t\(try Hardware.string(defaultID, kAudioDevicePropertyDeviceUID))")
                for device in try Hardware.outputs() {
                    print("OUTPUT\t\(device.id)\t\(device.uid)\t\(device.name)\t\(device.sampleRate)")
                }
                for app in try Hardware.applications() {
                    print("APP\t\(app.id)\t\(app.name)\t\(app.processes)")
                }
                return
            }
            if arguments.count == 5, arguments[1] == "--probe",
               let pid = pid_t(arguments[2]), let seconds = Double(arguments[4]), seconds > 0, seconds <= 60 {
                let processes = try Hardware.objects(Hardware.system, kAudioHardwarePropertyProcessObjectList)
                    .filter { (try? Hardware.scalar($0, kAudioProcessPropertyPID, initial: pid_t(0))) == pid }
                guard !processes.isEmpty else { throw AudioFailure(message: "Process has no audio object. Start playback first.") }
                guard let output = try Hardware.outputs().first(where: { $0.uid == arguments[3] }) else {
                    throw AudioFailure(message: "Output UID is not connected.")
                }
                let route = try AudioRoute(processes: processes, device: output) { fputs("\($0)\n", stderr) }
                RunLoop.current.run(until: Date().addingTimeInterval(seconds))
                let frames = route.frames
                let nonzero = route.nonzeroFrames
                let resources = route.resourceIDs
                route.stop()
                print("PROBE\tframes=\(frames)\tnonzero=\(nonzero)\toutput=\(output.name)")
                RunLoop.current.run(until: Date().addingTimeInterval(0.25))
                let remaining = try Hardware.objects(Hardware.system, kAudioHardwarePropertyTapList)
                    + Hardware.objects(Hardware.system, kAudioHardwarePropertyDevices)
                for resource in resources {
                    guard !remaining.contains(resource) else {
                        throw AudioFailure(message: "Route resource \(resource) survived teardown.")
                    }
                }
                print("CLEANUP\tprivate tap and aggregate destroyed")
                guard frames > 0, nonzero > 0 else {
                    throw AudioFailure(message: "No nonzero audio reached the renderer. Check capture permission and source playback.")
                }
                return
            }
            AudioOutputApp.main()
        } catch {
            fputs("\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
