#define _CRT_SECURE_NO_WARNINGS

#include "rhythm_map.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct WavPcm {
    float *samples;
    size_t sample_count;
    uint32_t sample_rate;
    uint16_t channels;
} WavPcm;

static uint16_t read_u16_le(FILE *file, int *ok) {
    uint8_t bytes[2];
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        *ok = 0;
        return 0;
    }
    return (uint16_t)(bytes[0] | ((uint16_t)bytes[1] << 8));
}

static uint32_t read_u32_le(FILE *file, int *ok) {
    uint8_t bytes[4];
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        *ok = 0;
        return 0;
    }
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int skip_bytes(FILE *file, uint32_t count) {
    return fseek(file, (long)(count + (count & 1U)), SEEK_CUR) == 0;
}

static int read_wav(const char *path, WavPcm *wav, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    char id[4];
    int ok = 1;
    uint16_t format = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;
    long data_offset = 0;

    memset(wav, 0, sizeof(*wav));
    if (file == NULL) {
        snprintf(error, error_size, "cannot open WAV file: %s", path);
        return 0;
    }
    if (fread(id, 1, 4, file) != 4 || memcmp(id, "RIFF", 4) != 0) {
        snprintf(error, error_size, "input is not a RIFF WAV file");
        fclose(file);
        return 0;
    }
    (void)read_u32_le(file, &ok);
    if (!ok || fread(id, 1, 4, file) != 4 || memcmp(id, "WAVE", 4) != 0) {
        snprintf(error, error_size, "input has no WAVE signature");
        fclose(file);
        return 0;
    }

    while (fread(id, 1, 4, file) == 4) {
        uint32_t chunk_size = read_u32_le(file, &ok);
        if (!ok) {
            break;
        }
        if (memcmp(id, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                snprintf(error, error_size, "invalid WAV fmt chunk");
                fclose(file);
                return 0;
            }
            format = read_u16_le(file, &ok);
            wav->channels = read_u16_le(file, &ok);
            wav->sample_rate = read_u32_le(file, &ok);
            (void)read_u32_le(file, &ok);
            (void)read_u16_le(file, &ok);
            bits_per_sample = read_u16_le(file, &ok);
            if (!ok || !skip_bytes(file, chunk_size - 16)) {
                break;
            }
        } else if (memcmp(id, "data", 4) == 0) {
            data_offset = ftell(file);
            data_size = chunk_size;
            if (!skip_bytes(file, chunk_size)) {
                break;
            }
        } else if (!skip_bytes(file, chunk_size)) {
            break;
        }
    }

    if (!ok || data_offset <= 0 || data_size == 0 || wav->sample_rate == 0 ||
        wav->channels == 0) {
        snprintf(error, error_size, "WAV is missing a usable fmt or data chunk");
        fclose(file);
        return 0;
    }
    if (!((format == 1 && bits_per_sample == 16) ||
          (format == 3 && bits_per_sample == 32))) {
        snprintf(error, error_size,
                 "WAV must contain PCM16 or IEEE-float32 samples");
        fclose(file);
        return 0;
    }

    {
        size_t bytes_per_sample = bits_per_sample / 8U;
        size_t index;
        if (data_size % bytes_per_sample != 0) {
            snprintf(error, error_size, "WAV data size is not sample-aligned");
            fclose(file);
            return 0;
        }
        wav->sample_count = data_size / bytes_per_sample;
        wav->samples = (float *)malloc(wav->sample_count * sizeof(float));
        if (wav->samples == NULL) {
            snprintf(error, error_size, "cannot allocate WAV sample buffer");
            fclose(file);
            return 0;
        }
        if (fseek(file, data_offset, SEEK_SET) != 0) {
            snprintf(error, error_size, "cannot seek to WAV sample data");
            free(wav->samples);
            fclose(file);
            return 0;
        }
        for (index = 0; index < wav->sample_count; ++index) {
            if (format == 1) {
                int16_t sample = (int16_t)read_u16_le(file, &ok);
                wav->samples[index] = (float)sample / 32768.0F;
            } else {
                uint32_t bits = read_u32_le(file, &ok);
                memcpy(&wav->samples[index], &bits, sizeof(bits));
            }
            if (!ok) {
                snprintf(error, error_size, "truncated WAV sample data");
                free(wav->samples);
                fclose(file);
                return 0;
            }
        }
    }
    fclose(file);
    return 1;
}

static void print_last_error(const char *operation) {
    const char *message = rhythm_map_last_error();
    fprintf(stderr, "%s: %s\n", operation,
            message != NULL && message[0] != '\0' ? message : "unknown error");
}

int main(int argc, char **argv) {
    RhythmMapAnalyzer *analyzer;
    WavPcm wav;
    char wav_error[256];
    char *json;

    if (argc == 2 && strcmp(argv[1], "--abi-only") == 0) {
        printf("Rhythm Map ABI %u\n", rhythm_map_abi_version());
        return rhythm_map_abi_version() == 1 ? 0 : 1;
    }
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s <model-pack.json> <model-directory> <audio.wav>\n",
                argv[0]);
        return 2;
    }
    if (!read_wav(argv[3], &wav, wav_error, sizeof(wav_error))) {
        fprintf(stderr, "decode WAV: %s\n", wav_error);
        return 1;
    }

    analyzer = rhythm_map_analyzer_new_from_model_pack(argv[1], argv[2]);
    if (analyzer == NULL) {
        print_last_error("create analyzer");
        free(wav.samples);
        return 1;
    }
    json = rhythm_map_analyze_pcm_json(analyzer, wav.samples, wav.sample_count,
                                       wav.sample_rate, wav.channels);
    free(wav.samples);
    if (json == NULL) {
        print_last_error("analyze PCM");
        rhythm_map_analyzer_free(analyzer);
        return 1;
    }

    puts(json);
    rhythm_map_string_free(json);
    rhythm_map_analyzer_free(analyzer);
    return 0;
}
