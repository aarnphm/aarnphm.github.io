#include "AudioRender.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct AORenderState {
    uint32_t inputOffset;
    _Atomic float gain;
    _Atomic uint64_t frames;
    _Atomic uint64_t nonzeroFrames;
};

AORenderState *ao_render_create(uint32_t inputOffset) {
    AORenderState *state = calloc(1, sizeof(*state));
    if (state) {
        state->inputOffset = inputOffset;
        atomic_init(&state->gain, 1.0f);
    }
    return state;
}
void ao_render_set_gain(AORenderState *state, float gain) {
    atomic_store_explicit(&state->gain, isfinite(gain) ? fminf(3, fmaxf(0, gain)) : 1, memory_order_relaxed);
}
void ao_render_destroy(AORenderState *state) { free(state); }
uint64_t ao_render_frames(AORenderState *state) {
    return atomic_load_explicit(&state->frames, memory_order_relaxed);
}
uint64_t ao_render_nonzero_frames(AORenderState *state) {
    return atomic_load_explicit(&state->nonzeroFrames, memory_order_relaxed);
}

OSStatus ao_render(AudioObjectID device, const AudioTimeStamp *now,
                   const AudioBufferList *input, const AudioTimeStamp *inputTime,
                   AudioBufferList *output, const AudioTimeStamp *outputTime, void *context) {
    AORenderState *state = context;
    if (!state || !input || !output) return noErr;
    for (UInt32 b = 0; b < output->mNumberBuffers; b++) {
        if (output->mBuffers[b].mData)
            memset(output->mBuffers[b].mData, 0, output->mBuffers[b].mDataByteSize);
    }
    const float *source[2] = {NULL, NULL};
    UInt32 stride[2] = {0, 0}, count[2] = {0, 0};
    UInt32 channel = 0;
    for (UInt32 b = 0; b < input->mNumberBuffers; b++) {
        const AudioBuffer *buffer = &input->mBuffers[b];
        for (UInt32 c = 0; c < buffer->mNumberChannels; c++, channel++) {
            if (channel < state->inputOffset || channel >= state->inputOffset + 2 || !buffer->mData) continue;
            UInt32 index = channel - state->inputOffset;
            source[index] = (const float *)buffer->mData + c;
            stride[index] = buffer->mNumberChannels;
            count[index] = buffer->mDataByteSize / (sizeof(float) * stride[index]);
        }
    }
    if (!source[0] || !source[1]) return noErr;
    UInt32 frames = count[0] < count[1] ? count[0] : count[1];
    UInt32 outputChannels = 0;
    for (UInt32 b = 0; b < output->mNumberBuffers; b++) outputChannels += output->mBuffers[b].mNumberChannels;
    channel = 0;
    UInt32 rendered = 0;
    const float gain = atomic_load_explicit(&state->gain, memory_order_relaxed);
    for (UInt32 b = 0; b < output->mNumberBuffers; b++) {
        AudioBuffer *buffer = &output->mBuffers[b];
        if (!buffer->mNumberChannels) continue;
        UInt32 available = buffer->mDataByteSize / (sizeof(float) * buffer->mNumberChannels);
        UInt32 n = frames < available ? frames : available;
        for (UInt32 c = 0; c < buffer->mNumberChannels; c++, channel++) {
            if (channel >= 2 || !buffer->mData) continue;
            float *destination = (float *)buffer->mData + c;
            for (UInt32 f = 0; f < n; f++) {
                float value = outputChannels == 1
                    ? (source[0][f * stride[0]] + source[1][f * stride[1]]) * 0.5f
                    : source[channel][f * stride[channel]];
                value = isfinite(value) ? value * gain : 0;
                destination[f * buffer->mNumberChannels] = fminf(1, fmaxf(-1, value));
            }
            rendered = n;
        }
    }
    uint64_t nonzero = 0;
    for (UInt32 f = 0; f < rendered; f++)
        if (fabsf(source[0][f * stride[0]]) > 0.00001f || fabsf(source[1][f * stride[1]]) > 0.00001f) nonzero++;
    atomic_fetch_add_explicit(&state->frames, rendered, memory_order_relaxed);
    atomic_fetch_add_explicit(&state->nonzeroFrames, nonzero, memory_order_relaxed);
    return noErr;
}
