using System;
using System.Runtime.InteropServices;
using System.Text;

namespace RhythmMap.Examples
{
    public sealed class RhythmMapException : Exception
    {
        public RhythmMapException(string message) : base(message) { }
    }

    public sealed class RhythmMapAnalyzer : SafeHandle
    {
        private const string LibraryName = "rhythm_map_ffi";

        private RhythmMapAnalyzer() : base(IntPtr.Zero, true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        public static uint AbiVersion() => NativeMethods.rhythm_map_abi_version();

        public static RhythmMapAnalyzer Create(string manifestPath, string modelDirectory)
        {
            RhythmMapAnalyzer analyzer = NativeMethods.rhythm_map_analyzer_new_from_model_pack(
                manifestPath,
                modelDirectory
            );
            if (analyzer.IsInvalid)
            {
                analyzer.Dispose();
                throw new RhythmMapException(LastError());
            }
            return analyzer;
        }

        public string AnalyzePcm(float[] interleavedSamples, uint sampleRate, ushort channels)
        {
            if (interleavedSamples == null)
            {
                throw new ArgumentNullException(nameof(interleavedSamples));
            }
            IntPtr result = NativeMethods.rhythm_map_analyze_pcm_json(
                this,
                interleavedSamples,
                (UIntPtr)(uint)interleavedSamples.Length,
                sampleRate,
                channels
            );
            if (result == IntPtr.Zero)
            {
                throw new RhythmMapException(LastError());
            }
            try
            {
                return Utf8String(result);
            }
            finally
            {
                NativeMethods.rhythm_map_string_free(result);
            }
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.rhythm_map_analyzer_free(handle);
            return true;
        }

        private static string LastError()
        {
            IntPtr pointer = NativeMethods.rhythm_map_last_error();
            return pointer == IntPtr.Zero ? "unknown native error" : Utf8String(pointer);
        }

        private static string Utf8String(IntPtr pointer)
        {
            int length = 0;
            while (Marshal.ReadByte(pointer, length) != 0)
            {
                checked
                {
                    ++length;
                }
            }
            byte[] bytes = new byte[length];
            Marshal.Copy(pointer, bytes, 0, length);
            return Encoding.UTF8.GetString(bytes);
        }

        private static class NativeMethods
        {
            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
            internal static extern uint rhythm_map_abi_version();

            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
            internal static extern RhythmMapAnalyzer rhythm_map_analyzer_new_from_model_pack(
                [MarshalAs(UnmanagedType.LPUTF8Str)] string manifestPath,
                [MarshalAs(UnmanagedType.LPUTF8Str)] string artifactRoot
            );

            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
            internal static extern IntPtr rhythm_map_analyze_pcm_json(
                RhythmMapAnalyzer analyzer,
                [In] float[] samples,
                UIntPtr sampleCount,
                uint sampleRate,
                ushort channels
            );

            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
            internal static extern IntPtr rhythm_map_last_error();

            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
            internal static extern void rhythm_map_string_free(IntPtr value);

            [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
            internal static extern void rhythm_map_analyzer_free(IntPtr analyzer);
        }
    }
}
