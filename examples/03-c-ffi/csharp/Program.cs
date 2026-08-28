using System.Buffers.Binary;
using System.Text;
using RhythmMap.Examples;

if (args is ["--abi-only"])
{
    Console.WriteLine($"Rhythm Map ABI {RhythmMapAnalyzer.AbiVersion()}");
    return;
}
if (args.Length != 3)
{
    Console.Error.WriteLine(
        "usage: RhythmMapExample <model-pack.json> <model-directory> <audio.wav>"
    );
    Environment.ExitCode = 2;
    return;
}

(float[] samples, uint sampleRate, ushort channels) = ReadPcm16Wav(args[2]);
using RhythmMapAnalyzer analyzer = RhythmMapAnalyzer.Create(args[0], args[1]);
Console.WriteLine(analyzer.AnalyzePcm(samples, sampleRate, channels));

static (float[] Samples, uint SampleRate, ushort Channels) ReadPcm16Wav(string path)
{
    using FileStream stream = File.OpenRead(path);
    using BinaryReader reader = new(stream, Encoding.ASCII, leaveOpen: true);
    if (Encoding.ASCII.GetString(reader.ReadBytes(4)) != "RIFF")
    {
        throw new InvalidDataException("input is not a RIFF WAV file");
    }
    _ = reader.ReadUInt32();
    if (Encoding.ASCII.GetString(reader.ReadBytes(4)) != "WAVE")
    {
        throw new InvalidDataException("input has no WAVE signature");
    }

    ushort format = 0;
    ushort channels = 0;
    uint sampleRate = 0;
    ushort bitsPerSample = 0;
    byte[]? data = null;
    while (stream.Position + 8 <= stream.Length)
    {
        string chunk = Encoding.ASCII.GetString(reader.ReadBytes(4));
        uint chunkSize = reader.ReadUInt32();
        long nextChunk = checked(stream.Position + chunkSize + (chunkSize & 1));
        if (nextChunk > stream.Length)
        {
            throw new InvalidDataException("truncated WAV chunk");
        }
        if (chunk == "fmt ")
        {
            if (chunkSize < 16)
            {
                throw new InvalidDataException("invalid WAV fmt chunk");
            }
            format = reader.ReadUInt16();
            channels = reader.ReadUInt16();
            sampleRate = reader.ReadUInt32();
            _ = reader.ReadUInt32();
            _ = reader.ReadUInt16();
            bitsPerSample = reader.ReadUInt16();
        }
        else if (chunk == "data")
        {
            data = reader.ReadBytes(checked((int)chunkSize));
        }
        stream.Position = nextChunk;
    }
    if (format != 1 || bitsPerSample != 16 || channels == 0 || sampleRate == 0 || data is null)
    {
        throw new InvalidDataException("example accepts uncompressed PCM16 WAV input");
    }
    if (data.Length % 2 != 0)
    {
        throw new InvalidDataException("WAV data size is not sample-aligned");
    }
    float[] samples = new float[data.Length / 2];
    for (int index = 0; index < samples.Length; ++index)
    {
        samples[index] = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(index * 2, 2)) / 32768.0f;
    }
    return (samples, sampleRate, channels);
}
