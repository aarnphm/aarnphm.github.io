import AppKit

guard CommandLine.arguments.count == 3,
      let sound = NSSound(contentsOfFile: CommandLine.arguments[1], byReference: false) else {
    fatalError("Usage: tone-fixture file.wav output-uid")
}
sound.playbackDeviceIdentifier = NSSound.PlaybackDeviceIdentifier(CommandLine.arguments[2])
sound.loops = true
guard sound.play() else { fatalError("Could not start fixture") }
print("Fixture playing on \(CommandLine.arguments[2])")
fflush(stdout)
RunLoop.current.run(until: Date().addingTimeInterval(30))
sound.stop()
