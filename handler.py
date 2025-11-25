import asyncio
import base64
import runpod
import os
import sys
from pathlib import Path

# --- FIX: Ensure project root is in python path ---
# This ensures 'voicevox_engine' package is found regardless of how the script is called
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# -------------------------------------------------

try:
    # Standard imports based on your project structure
    from voicevox_engine.tts_pipeline.tts_engine import TTSEngine
    from voicevox_engine.core.core_wrapper import CoreWrapper
    from voicevox_engine.preset.preset_manager import PresetManager

    # Use UserDictionary if UserDictManager is not available (API change in recent versions)
    try:
        from voicevox_engine.user_dict.user_dict_manager import UserDictManager
    except ImportError:
        # Fallback: In some versions it might be named differently or located elsewhere
        # Based on your context, it seems user_dict_manager.py exists, but let's be safe
        print("Warning: UserDictManager import failed, trying alternate import...")
        from voicevox_engine.user_dict.user_dict_manager import UserDictionary as UserDictManager

    from voicevox_engine.model import AudioQuery
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Current directory contents: {os.listdir('.')}")
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
    # We explicitly set use_gpu=True.
    core = CoreWrapper(
        use_gpu=True,
        core_dir=core_dir,
        voicelib_dir=None,
        # runtime_dir=None, # Removed as it might not be in __init__ args for this version
        load_all_models=True  # Loading all models to prevent runtime delay
    )

    # Initialize the core is handled inside __init__ of CoreWrapper usually,
    # but we check if we need manual init for specific versions.
    # core.initialize(use_gpu=True) # Often called inside CoreWrapper __init__

    # 3. Initialize Managers
    # Try initializing UserDictManager (or UserDictionary)
    try:
        user_dict_manager = UserDictManager()
    except TypeError:
        # Newer versions of UserDictManager might require args or act differently
        # If it's actually UserDictionary class, it might exist without manager wrapper
        user_dict_manager = UserDictManager()

        # Preset Manager
    # Check if we need to pass a path or if it handles defaults
    preset_path = root_dir / "presets.yaml"
    if preset_path.exists():
        preset_manager = PresetManager(preset_path=preset_path)
    else:
        # Create a dummy path if needed or pass None if supported
        # Based on source, it expects a Path object
        preset_manager = PresetManager(preset_path=Path("presets.yaml"))

    # 4. Initialize TTSEngine
    tts_engine = TTSEngine(
        core=core,  # Note: arg name is 'core' in TTSEngine __init__ snippet provided
        # user_dict_manager=user_dict_manager, # TTSEngine might not take these in __init__ directly depending on version
        # preset_manager=preset_manager
    )

    # Inject managers if they are attached properties (common in dependency injection patterns)
    # or if the TTSEngine expects them in a specific way.
    # Based on your provided TTSEngine class, it only takes `core: CoreWrapper` in __init__
    # So we just pass core.

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
        # create_audio_query or similar method
        # The TTSEngine you shared has `create_accent_phrases` but usually high level has `compute_audio_query`
        # If high level method isn't exposed in TTSEngine, we construct AudioQuery manually
        # using the helper methods shown in your context (create_accent_phrases)

        # Logic mimicking `run.py` or standard API:
        # A. Create Accent Phrases
        accent_phrases = tts_engine.create_accent_phrases(
            text=text,
            style_id=speaker_id,
            enable_katakana_english=job_input.get("enable_katakana_english", True)
        )

        # B. Construct AudioQuery
        # We need default values for the other fields
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
        # wave_data is numpy array (float32). We need to convert to 16-bit PCM wav bytes
        import numpy as np
        import io
        import soundfile as sf

        # Create in-memory wav file
        buffer = io.BytesIO()
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
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    # Allow running locally for testing
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(handler({"input": {"text": "テストです"}}))
    else:
        runpod.serverless.start({"handler": handler})