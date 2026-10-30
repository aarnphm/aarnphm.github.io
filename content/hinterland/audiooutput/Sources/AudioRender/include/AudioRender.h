#pragma once
#include <CoreAudio/CoreAudio.h>
#include <stdint.h>

typedef struct AORenderState AORenderState;
AORenderState * _Nullable ao_render_create(uint32_t inputOffset);
void ao_render_set_gain(AORenderState * _Nonnull state, float gain);
void ao_render_destroy(AORenderState * _Nonnull state);
uint64_t ao_render_frames(AORenderState * _Nonnull state);
uint64_t ao_render_nonzero_frames(AORenderState * _Nonnull state);
OSStatus ao_render(AudioObjectID device, const AudioTimeStamp * _Nonnull now,
                   const AudioBufferList * _Nonnull input, const AudioTimeStamp * _Nonnull inputTime,
                   AudioBufferList * _Nonnull output, const AudioTimeStamp * _Nonnull outputTime, void * _Nullable context);
OSStatus ao_select_tap_input(AudioObjectID device, AudioDeviceIOProcID _Nonnull proc, uint32_t firstTapChannel);
