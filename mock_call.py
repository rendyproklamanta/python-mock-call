import sounddevice as sd
import requests
import io
import wave
import numpy as np
import time
from collections import deque

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

def start_mock_call(phone,
                    samplerate=16000,
                    device=0,
                    frame_duration=0.1,
                    energy_threshold=0.01,
                    min_speech_duration=0.2,
                    silence_duration=1.0,
                    gain=1.0):
    
    # Do not override global default device (avoid assigning an input-only device as default output)
    # Pass `device` explicitly to the InputStream below.
    block_frames = int(frame_duration * samplerate)

    print("🎤 Siap merekam — bicara kapan saja. Tekan Ctrl+C untuk berhenti.")

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', device=device) as stream:
            is_recording = False
            recorded_frames = []
            speech_start_time = None
            last_voice_time = None

            while True:
                data, _ = stream.read(block_frames)
                # data shape: (block_frames, 1)
                mono = data.flatten()
                rms = np.sqrt(np.mean(mono.astype('float64')**2))

                now = time.monotonic()

                if not is_recording:
                    # waiting for voice
                    if rms >= energy_threshold:
                        is_recording = True
                        recorded_frames = [mono.copy()]
                        speech_start_time = now
                        last_voice_time = now
                        print("🔴 Voice detected — mulai rekam")
                    # else keep waiting
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

                                # If server returned an audio file, play it directly from memory
                                ctype = response.headers.get('content-type', '')
                                if response.status_code == 200 and (ctype.startswith('audio') or 'wav' in ctype):
                                    audio_bytes = response.content
                                    bio = io.BytesIO(audio_bytes)
                                    try:
                                        with wave.open(bio, 'rb') as wf_resp:
                                            nchannels = wf_resp.getnchannels()
                                            sampwidth = wf_resp.getsampwidth()
                                            framerate = wf_resp.getframerate()
                                            nframes = wf_resp.getnframes()
                                            frames = wf_resp.readframes(nframes)

                                        # Convert to float32 in range [-1, 1]
                                        if sampwidth == 1:
                                            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                                            data = (data - 128) / 128.0
                                        elif sampwidth == 2:
                                            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                                            data = data / 32767.0
                                        else:
                                            data = None

                                        if data is None:
                                            # unsupported sample width: save to disk
                                            with open('response_audio.wav', 'wb') as f:
                                                f.write(audio_bytes)
                                            print('📥 Server returned audio but sampwidth not supported; saved to response_audio.wav')
                                        else:
                                            if nchannels > 1:
                                                data = data.reshape(-1, nchannels)

                                            # Robust playback: query default output device channels and adapt
                                            try:
                                                out_info = sd.query_devices(kind='output')
                                                out_channels = out_info.get('max_output_channels', 1)
                                            except Exception:
                                                out_channels = 1

                                            # Ensure data has shape (frames,) for mono or (frames, channels)
                                            if data.ndim == 1:
                                                if out_channels == 1:
                                                    play_data = data
                                                else:
                                                    # duplicate mono to match output channels
                                                    play_data = np.tile(data.reshape(-1, 1), (1, out_channels))
                                            else:
                                                # multi-channel data
                                                if data.shape[1] < out_channels:
                                                    # repeat channels to fill
                                                    reps = int(np.ceil(out_channels / data.shape[1]))
                                                    play_data = np.tile(data, (1, reps))[:, :out_channels]
                                                else:
                                                    play_data = data[:, :out_channels]

                                            print('▶️ Playing server audio response...')
                                            try:
                                                sd.play(play_data, framerate)
                                                sd.wait()
                                            except Exception as e:
                                                # Often fails if chosen device is input-only or channels mismatch
                                                print(f"❌ Playback failed: {e}; saved server audio to response_audio.wav")
                                                with open('response_audio.wav', 'wb') as f:
                                                    f.write(audio_bytes)
                                    except wave.Error:
                                        # not a valid WAV, save to file
                                        with open('response_audio.wav', 'wb') as f:
                                            f.write(audio_bytes)
                                        print('📥 Server returned audio but it was not a valid WAV; saved to response_audio.wav')
                                else:
                                    # non-audio response
                                    try:
                                        print(f"📞 Response dari server: {response.text}")
                                    except Exception:
                                        print('📞 Response diterima tetapi tidak dapat ditampilkan sebagai teks')
                        except requests.exceptions.RequestException as e:
                            print(f"❌ Error mengirim ke server: {e}")

                        # reset state and wait for next speech
                        is_recording = False
                        recorded_frames = []
                        speech_start_time = None
                        last_voice_time = None
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