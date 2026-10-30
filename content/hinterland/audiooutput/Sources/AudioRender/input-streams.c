#include "AudioRender.h"
#include <stdlib.h>

OSStatus ao_select_tap_input(AudioObjectID device, AudioDeviceIOProcID proc, uint32_t firstTapChannel) {
    AudioObjectPropertyAddress address = {kAudioDevicePropertyStreams, kAudioObjectPropertyScopeInput, kAudioObjectPropertyElementMain};
    UInt32 size = 0;
    OSStatus status = AudioObjectGetPropertyDataSize(device, &address, 0, NULL, &size);
    if (status != noErr) return status;
    UInt32 count = size / sizeof(AudioObjectID);
    if (!count) return kAudioHardwareBadStreamError;
    AudioObjectID *streams = malloc(size);
    if (!streams) return kAudioHardwareUnspecifiedError;
    status = AudioObjectGetPropertyData(device, &address, 0, NULL, &size, streams);
    UInt32 usageSize = sizeof(AudioHardwareIOProcStreamUsage) + (count - 1) * sizeof(UInt32);
    AudioHardwareIOProcStreamUsage *usage = calloc(1, usageSize);
    if (!usage) { free(streams); return kAudioHardwareUnspecifiedError; }
    usage->mIOProc = (void *)proc;
    usage->mNumberStreams = count;
    for (UInt32 index = 0; status == noErr && index < count; index++) {
        UInt32 start = 0, startSize = sizeof(start);
        AudioObjectPropertyAddress channel = {kAudioStreamPropertyStartingChannel, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(streams[index], &channel, 0, NULL, &startSize, &start);
        usage->mStreamIsOn[index] = start > firstTapChannel;
    }
    if (status == noErr) {
        address.mSelector = kAudioDevicePropertyIOProcStreamUsage;
        status = AudioObjectSetPropertyData(device, &address, 0, NULL, usageSize, usage);
    }
    free(usage);
    free(streams);
    return status;
}
