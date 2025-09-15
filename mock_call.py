import sounddevice as sd
import requests
import io
import wave
import numpy as np
import time
from collections import deque
from pydub import AudioSegment

def play_response_audio(audio_bytes):
    """Play server audio (WAV/MP3/OGG) dynamically"""
    import io
    import wave
    import numpy as np

    bio = io.BytesIO(audio_bytes)

    try:
        # --- coba baca sebagai WAV ---
        with wave.open(bio, 'rb') as wf_resp:
            nchannels = wf_resp.getnchannels()
            sampwidth = wf_resp.getsampwidth()
            framerate = wf_resp.getframerate()
            nframes = wf_resp.getnframes()
            frames = wf_resp.readframes(nframes)

        if sampwidth == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128) / 128.0
        elif sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            raise ValueError("Unsupported sample width")

        if nchannels > 1:
            data = data.reshape(-1, nchannels)

        print("▶️ Playing WAV response...")
        sd.play(data, framerate)
        sd.wait()
        return

    except Exception:
        # --- fallback ke pydub (auto deteksi format: mp3/ogg/wav/...) ---
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            samples = np.array(seg.get_array_of_samples()).astype(np.float32) / (1 << 15)
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels)
            print(f"▶️ Playing non-WAV (detected {seg.frame_rate} Hz, {seg.channels} ch)...")
            sd.play(samples, seg.frame_rate)
            sd.wait()
        except Exception as e:
            with open("response_audio_unknown.bin", "wb") as f:
                f.write(audio_bytes)
            print(f"❌ Tidak bisa memutar audio. Disimpan ke response_audio_unknown.bin ({e})")



def normalize_audio(audio, target_level=0.8, gain=1.0):
    """Simple audio normalization with gain"""
    # Find the maximum absolute value
    max_val = np.max(np.abs(audio))
    
    # Avoid division by zero
    if max_val > 0:
        # Normalize to target level (0.8 = 80% of max)
        normalized = audio * (target_level / max_val)
    else:
        normalized = audio
    
    # Apply gain
    normalized = normalized * gain
    
    # Clip to [-1, 1] to avoid overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized

def detect_input_device(samplerate):
    """Return an input-capable device index or None if not found."""
    try:
        default = sd.default.device
    except Exception:
        default = None

    # If a default input device is set and has input channels, prefer it
    if default:
        dev_idx = default[0] if isinstance(default, (list, tuple)) else default
        try:
            dev = sd.query_devices(dev_idx)
            if dev.get('max_input_channels', 0) > 0:
                # verify samplerate compatibility
                try:
                    sd.check_input_settings(device=dev_idx, samplerate=samplerate)
                    return dev_idx
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: pick the first device that has input channels and supports the samplerate
    for i, dev in enumerate(sd.query_devices()):
        if dev.get('max_input_channels', 0) > 0:
            try:
                sd.check_input_settings(device=i, samplerate=samplerate)
                return i
            except Exception:
                continue

    return None

def start_mock_call(phone,
                    samplerate=16000,
                    device=None,
                    frame_duration=0.1,
                    energy_threshold=0.05,
                    trigger_duration=0.15,           # require this many seconds of energy before start
                    min_speech_duration=0.2,
                    silence_duration=1.0,
                    gain=1.0):
    
    # Do not override global default device (avoid assigning an input-only device as default output)
    # Pass `device` explicitly to the InputStream below.
    block_frames = int(frame_duration * samplerate)
    required_consecutive = max(1, int(trigger_duration / frame_duration))

    # Auto-detect input device if not provided
    if device is None:
        device = detect_input_device(samplerate)
        
        if device is None:
            raise RuntimeError("No input-capable audio device found. Check your audio devices.")
        try:
            dev_info = sd.query_devices(device)
            print(f"Using input device #{device}: {dev_info.get('name')}")
        except Exception:
            print(f"Using input device #{device}")

    print("🎤 Siap merekam — bicara kapan saja. Tekan Ctrl+C untuk berhenti.")

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', device=device) as stream:
            is_recording = False
            recorded_frames = []
            speech_start_time = None
            last_voice_time = None
            # explicit voice frame counter and post-record cooldown
            voice_frame_count = 0
            cooldown_until = 0.0

            while True:
                data, _ = stream.read(block_frames)
                # data shape: (block_frames, 1)
                mono = data.flatten()
                # remove per-frame DC offset to reduce tiny-bias triggers
                mono = mono - np.mean(mono)
                rms = np.sqrt(np.mean(mono.astype('float64')**2))

                now = time.monotonic()
                
                # short cooldown after finishing a recording to avoid immediate retrigger
                if now < cooldown_until:
                    continue

                if not is_recording:
                    # waiting for voice
                    if rms >= energy_threshold:
                        voice_frame_count += 1
                    else:
                        voice_frame_count = 0

                    if voice_frame_count >= required_consecutive:
                        is_recording = True
                        recorded_frames = [mono.copy()]
                        speech_start_time = now
                        last_voice_time = now
                        voice_frame_count = 0
                        print("🔴 Voice detected — mulai rekam")
                else:
                    recorded_frames.append(mono.copy())
                    if rms >= energy_threshold:
                        last_voice_time = now
                    # check for end of speech
                    silence_elapsed = now - (last_voice_time or now)
                    speech_elapsed = now - (speech_start_time or now)
                    if silence_elapsed >= silence_duration and speech_elapsed >= min_speech_duration:
                        # finalize
                        audio = np.concatenate(recorded_frames)
                        print(f"⏹️ Selesai merekam ({speech_elapsed:.2f}s). RMS preview: {rms:.6f}")

                        # Normalize and save
                        normalized_audio = normalize_audio(audio, gain=gain)
                        audio_int16 = (normalized_audio * 32767).astype(np.int16)
                        with wave.open("output.wav", 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(samplerate)
                            wf.writeframes(audio_int16.tobytes())

                        # Send to server
                        print(f"=> Kirim payload dari nomor: {phone}")
                        try:
                            with open('output.wav', 'rb') as audio_file:
                                files = {
                                    'phone': (None, phone),
                                    'audio': ('output.wav', audio_file, 'audio/wav')
                                }
                                response = requests.post("http://localhost:8000/call/mock", files=files)

                                # If server returned an audio file, play it
                                ctype = response.headers.get('content-type', '')
                                if response.status_code == 200 and ctype.startswith('audio'):
                                    play_response_audio(response.content)
                                else:
                                    try:
                                        print(f"📞 Response dari server: {response.text}")
                                    except Exception:
                                        print("📞 Response diterima tapi tidak bisa ditampilkan sebagai teks")
                        except requests.exceptions.RequestException as e:
                            print(f"❌ Error mengirim ke server: {e}")

                        # reset state and wait for next speech
                        is_recording = False
                        recorded_frames = []
                        speech_start_time = None
                        last_voice_time = None
                        voice_frame_count = 0
                        # set brief cooldown to avoid retrigger from buffered/residual audio
                        cooldown_until = time.monotonic() + max(0.2, frame_duration * required_consecutive)
                        print("Siap untuk rekaman berikutnya...")

    except KeyboardInterrupt:
        print("\n❌ Call berakhir.")
    except Exception as e:
        print(f"❌ Error dalam recording: {e}")

if __name__ == "__main__":
    # List available audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print("\n" + "="*50 + "\n")
    
    start_mock_call("123456789")