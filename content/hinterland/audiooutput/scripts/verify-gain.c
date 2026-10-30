// Exercise the production callback: gain, mute, clipping, invalid samples and mono output.
#include "AudioRender.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
    AORenderState *state = ao_render_create(0);
    assert(state);
    float samples[] = {0.1f, -0.2f, 0.8f, -0.9f, NAN, INFINITY};
    float result[6] = {0};
    AudioBufferList input = {1, {{2, sizeof(samples), samples}}};
    AudioBufferList output = {1, {{2, sizeof(result), result}}};
    AudioTimeStamp time = {0};
    ao_render(0, &time, &input, &time, &output, &time, state);
    assert(fabsf(result[0] - 0.1f) < 0.0001f);
    ao_render_set_gain(state, 3);
    ao_render(0, &time, &input, &time, &output, &time, state);
    assert(fabsf(result[0] - 0.3f) < 0.0001f);
    assert(fabsf(result[1] + 0.6f) < 0.0001f);
    assert(result[2] == 1 && result[3] == -1);
    assert(result[4] == 0 && result[5] == 0);
    ao_render_set_gain(state, 0);
    ao_render(0, &time, &input, &time, &output, &time, state);
    for (int i = 0; i < 6; i++) assert(result[i] == 0);
    ao_render_set_gain(state, 0.5f);
    output.mBuffers[0].mNumberChannels = 1;
    ao_render(0, &time, &input, &time, &output, &time, state);
    assert(fabsf(result[0] + 0.025f) < 0.0001f);
    ao_render_set_gain(state, NAN);
    ao_render(0, &time, &input, &time, &output, &time, state);
    assert(fabsf(result[0] + 0.05f) < 0.0001f);
    ao_render_destroy(state);
    puts("PASS: unity, 300%, clipping, nonfinite samples, mute, mono, invalid gain");
}
