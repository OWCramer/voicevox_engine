import asyncio
import base64
import runpod
import os
import sys
from pathlib import Path

# Import VOICEVOX Engine components
from voicevox_engine.tts_pipeline.tts_engine import TTSEngine
from voicevox_engine.core.core_wrapper import CoreWrapper
from voicevox_engine.preset.preset_manager import PresetManager
from voicevox_engine.user_dict.user_dict_manager import UserDictManager

# Global engine instance
tts_engine = None


async def initialize_engine():
    """Initializes the VOICEVOX engine with GPU support."""
    global tts_engine
    if tts_engine is not None:
        return tts_engine

    print("--- Initializing VOICEVOX Engine (GPU) ---")

    # 1. Define Core Paths
    # The Dockerfile puts the core libraries in /app/voicevox_core
    root_dir = Path("/app")
    core_dir = root_dir / "voicevox_core"

    if not core_dir.exists():
        # Fallback for local testing if not in /app
        core_dir = Path("voicevox_core")

    print(f"Core directory: {core_dir}")

    # 2. Initialize CoreWrapper
    # We explicitly set use_gpu=True.
    # The LD_LIBRARY_PATH in Dockerfile ensures the .so files are found.
    core = CoreWrapper(
        use_gpu=True,
        voicevox_dir=core_dir,
        voicelib_dir=None,
        runtime_dir=None,
        enable_mock=False
    )

    # Initialize the core (this loads the C++ libraries)
    if not core.is_initialized():
        print("Loading Core libraries...")
        core.initialize(use_gpu=True)

    print(f"Core initialized. GPU Enabled: {core.use_gpu}")

    # 3. Initialize Managers
    # User Dictionary Manager
    user_dict_manager = UserDictManager()

    # Preset Manager (loads default presets or from file)
    preset_manager = PresetManager(preset_file_path=None)
    try:
        # Try to load presets if the file exists, otherwise ignore
        preset_path = root_dir / "presets.yaml"
        if preset_path.exists():
            preset_manager = PresetManager(preset_file_path=preset_path)
    except Exception as e:
        print(f"Warning: Could not load presets: {e}")

    # 4. Initialize TTSEngine
    tts_engine = TTSEngine(
        core_wrapper=core,
        user_dict_manager=user_dict_manager,
        preset_manager=preset_manager
    )

    print("--- Engine Initialization Complete ---")
    return tts_engine


async def handler(job):
    """
    RunPod Handler
    Input: {"input": {"text": "...", "speaker_id": 1, ...}}
    """
    global tts_engine

    # Warm up engine if needed
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
        # Generate the audio query from text
        audio_query = await tts_engine.compute_audio_query(
            text=text,
            style_id=speaker_id,
            theme=None
        )

        # 2. Apply Parameters
        # Override standard query parameters with input values
        if "speed_scale" in job_input:
            audio_query.speedScale = float(job_input["speed_scale"])
        if "pitch_scale" in job_input:
            audio_query.pitchScale = float(job_input["pitch_scale"])
        if "intonation_scale" in job_input:
            audio_query.intonationScale = float(job_input["intonation_scale"])
        if "volume_scale" in job_input:
            audio_query.volumeScale = float(job_input["volume_scale"])
        if "pre_phoneme_length" in job_input:
            audio_query.prePhonemeLength = float(job_input["pre_phoneme_length"])
        if "post_phoneme_length" in job_input:
            audio_query.postPhonemeLength = float(job_input["post_phoneme_length"])

        # Support for simple kana input (if pre-processed)
        # Note: standard compute_audio_query handles raw text.
        # If you want direct Kana injection, you'd modify the query.kana here.

        # 3. Synthesis
        # Generate wav data
        print(f"Synthesizing: '{text}' (Speaker: {speaker_id})")

        wave_data = await tts_engine.synthesis(
            query=audio_query,
            style_id=speaker_id,
            enable_interrogative_upspeak=job_input.get("enable_interrogative_upspeak", True)
        )

        # 4. Return Result
        # RunPod expects JSON. We encode binary audio to base64.
        base64_audio = base64.b64encode(wave_data).decode("utf-8")

        return {
            "audio_base64": base64_audio,
            "sampling_rate": 24000,  # Standard Voicevox rate
            "status": "success"
        }

    except Exception as e:
        print(f"Error processing request: {e}")
        # Print full traceback for logs
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


# Start the handler
runpod.serverless.start({"handler": handler})