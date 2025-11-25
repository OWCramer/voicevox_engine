import asyncio
import base64
import runpod
import os
import sys
from pathlib import Path
import traceback

# --- FIX: Ensure project root is in python path ---
# This ensures 'voicevox_engine' package is found regardless of how the script is called
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# -------------------------------------------------

try:
    from voicevox_engine.tts_pipeline.tts_engine import TTSEngine
    from voicevox_engine.core.core_wrapper import CoreWrapper
    from voicevox_engine.preset.preset_manager import PresetManager

    # Robust import for UserDictManager
    try:
        from voicevox_engine.user_dict.user_dict_manager import UserDictManager
    except ImportError:
        print("Warning: UserDictManager import failed, checking for UserDictionary...")
        try:
            from voicevox_engine.user_dict.user_dict_manager import UserDictionary as UserDictManager
        except ImportError:
            # Fallback if UserDictionary is also not importable directly
            print("Warning: UserDictionary import failed too.")
            UserDictManager = None

    from voicevox_engine.model import AudioQuery
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    traceback.print_exc()
    raise e

# Global engine instance
tts_engine = None


async def initialize_engine():
    """Initializes the VOICEVOX engine with GPU support."""
    global tts_engine
    if tts_engine is not None:
        return tts_engine

    print("--- Initializing VOICEVOX Engine (GPU) ---")

    # 1. Define Core Paths
    root_dir = Path("/app")
    core_dir = root_dir / "voicevox_core"

    if not core_dir.exists():
        # Fallback for local testing
        core_dir = Path("voicevox_core")

    print(f"Core directory: {core_dir}")

    # 2. Initialize CoreWrapper
    # FIX: Removing invalid arguments 'voicelib_dir'.
    # CoreWrapper only takes use_gpu, core_dir, cpu_num_threads, load_all_models
    core = CoreWrapper(
        use_gpu=True,
        core_dir=core_dir,
        cpu_num_threads=4,
        load_all_models=True
    )

    # Note: core.initialize() is called inside CoreWrapper.__init__ in this version,
    # so we don't need to call it manually unless we suppressed it.

    # 3. Initialize Managers
    user_dict_manager = None
    if UserDictManager:
        try:
            user_dict_manager = UserDictManager()
        except Exception as e:
            print(f"Warning: Failed to init UserDictManager: {e}")

    # Preset Manager
    preset_path = root_dir / "presets.yaml"
    if preset_path.exists():
        preset_manager = PresetManager(preset_path=preset_path)
    else:
        # Pass a non-existent path to init default empty presets if needed
        preset_manager = PresetManager(preset_path=Path("presets.yaml"))

    # 4. Initialize TTSEngine
    # TTSEngine in your version takes only `core` in __init__
    tts_engine = TTSEngine(
        core=core
    )

    # If the engine needs managers injected separately (some versions do)
    # tts_engine.user_dict_manager = user_dict_manager
    # tts_engine.preset_manager = preset_manager

    print("--- Engine Initialization Complete ---")
    return tts_engine


async def handler(job):
    """
    RunPod Handler
    Input: {"input": {"text": "...", "speaker_id": 1, ...}}
    """
    global tts_engine

    # Warm up engine
    if tts_engine is None:
        await initialize_engine()

    job_input = job.get("input", {})

    # Validate Input
    text = job_input.get("text")
    if not text:
        return {"error": "Missing 'text' in input"}

    speaker_id = int(job_input.get("speaker_id", 1))

    try:
        # 1. Audio Query
        # Mimicking run.py logic

        # A. Create Accent Phrases
        # TTSEngine.create_accent_phrases signature: (text, style_id, enable_katakana_english)
        accent_phrases = tts_engine.create_accent_phrases(
            text=text,
            style_id=speaker_id,
            enable_katakana_english=job_input.get("enable_katakana_english", True)
        )

        # B. Construct AudioQuery
        # Default values similar to standard engine defaults
        audio_query = AudioQuery(
            accent_phrases=accent_phrases,
            speedScale=float(job_input.get("speed_scale", 1.0)),
            pitchScale=float(job_input.get("pitch_scale", 0.0)),
            intonationScale=float(job_input.get("intonation_scale", 1.0)),
            volumeScale=float(job_input.get("volume_scale", 1.0)),
            prePhonemeLength=float(job_input.get("pre_phoneme_length", 0.1)),
            postPhonemeLength=float(job_input.get("post_phoneme_length", 0.1)),
            outputSamplingRate=24000,
            outputStereo=False,
        )

        # 2. Synthesis
        print(f"Synthesizing: '{text}' (Speaker: {speaker_id})")

        wave_data = tts_engine.synthesize_wave(
            query=audio_query,
            style_id=speaker_id,
            enable_interrogative_upspeak=job_input.get("enable_interrogative_upspeak", True)
        )

        # 3. Return Result
        # Convert float32 numpy array to WAV bytes
        import io
        import soundfile as sf

        buffer = io.BytesIO()
        # Voicevox output is typically 24k sample rate
        sf.write(buffer, wave_data, 24000, format='WAV', subtype='PCM_16')
        wav_bytes = buffer.getvalue()

        base64_audio = base64.b64encode(wav_bytes).decode("utf-8")

        return {
            "audio_base64": base64_audio,
            "sampling_rate": 24000,
            "status": "success"
        }

    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    # Allow running locally for testing
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(handler({"input": {"text": "テストです"}}))
    else:
        runpod.serverless.start({"handler": handler})