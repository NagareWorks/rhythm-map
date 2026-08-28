#if UNITY_2021_3_OR_NEWER
using System.IO;
using RhythmMap.Examples;
using UnityEngine;

public sealed class UnityAudioClipExample : MonoBehaviour
{
    [SerializeField] private AudioClip clip;
    [SerializeField] private string modelPack = "models/beat-this-full-v1.json";
    [SerializeField] private string modelDirectory = "models/beat-this-full-v1";

    private void Start()
    {
        // The clip must be readable (for example, Decompress On Load), because
        // AudioClip.GetData supplies the interleaved float PCM owned by Unity.
        float[] samples = new float[clip.samples * clip.channels];
        if (!clip.GetData(samples, 0))
        {
            Debug.LogError("AudioClip PCM is not readable");
            return;
        }

        string manifest = Path.Combine(Application.streamingAssetsPath, modelPack);
        string artifacts = Path.Combine(Application.streamingAssetsPath, modelDirectory);
        using RhythmMapAnalyzer analyzer = RhythmMapAnalyzer.Create(manifest, artifacts);
        string analysisJson = analyzer.AnalyzePcm(
            samples,
            checked((uint)clip.frequency),
            checked((ushort)clip.channels)
        );
        Debug.Log(analysisJson);
    }
}
#endif
