#ifndef RHYTHM_MAP_H
#define RHYTHM_MAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct RhythmMapAnalyzer RhythmMapAnalyzer;

uint32_t rhythm_map_abi_version(void);

RhythmMapAnalyzer *rhythm_map_analyzer_new(
    const char *mel_model_path,
    const char *beat_model_path);

/* Preferred constructor: verifies every artifact and records pack provenance. */
RhythmMapAnalyzer *rhythm_map_analyzer_new_from_model_pack(
    const char *manifest_path,
    const char *artifact_root);

char *rhythm_map_analyze_pcm_json(
    RhythmMapAnalyzer *analyzer,
    const float *samples,
    size_t sample_count,
    uint32_t sample_rate,
    uint16_t channels);

const char *rhythm_map_last_error(void);
void rhythm_map_string_free(char *value);
void rhythm_map_analyzer_free(RhythmMapAnalyzer *analyzer);

#ifdef __cplusplus
}
#endif

#endif
