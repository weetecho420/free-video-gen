"""
Free AI Video Studio
  Tab 1: Talking character  – picture + script -> voice (English/Tagalog) -> lip-synced video
  Tab 2: Cinematic talking  – picture + motion + script -> moving video with lip-sync
  Tab 3: Video + voice-over – picture + motion prompt + script -> moving video with narrator
  Tab 4: Silent video       – picture + motion prompt -> moving video
"""
import time

import gradio as gr

from voice import VOICES, DEFAULT_VOICE, make_voice, audio_seconds

VOICE_NAMES = list(VOICES.keys())


def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")


# ---------- Tab 1: talking character ----------
def talking(image, script, voice, speed, own_audio, full_image, enhance, size, expression):
    from talking import make_talking_video

    if image is None:
        raise gr.Error("Please upload a picture with a clear face.")
    if own_audio:
        audio = own_audio
    else:
        if not script or not script.strip():
            raise gr.Error("Type the script (what the character will say), or upload your own voice.")
        audio = make_voice(script, voice, out_wav=f"voice_{_stamp()}.wav", speed=speed)

    secs = audio_seconds(audio)
    video = make_talking_video(
        image, audio, full_image=full_image, enhance_face=enhance,
        size=int(size), expression=float(expression),
    )
    return audio, video, f"✅ Done! Voice length: {secs:.1f} seconds."


# ---------- Tab: cinematic talking (moving body + lip-sync) ----------
def cinematic(image, motion_prompt, script, voice, speed, own_audio, add_talking, sharpen):
    import math
    from generate import generate
    from lipsync import make_lipsync_video

    if image is None:
        raise gr.Error("Please upload a picture with a clear face.")
    if not motion_prompt or not motion_prompt.strip():
        raise gr.Error("Describe the movement / scene in the motion prompt.")
    tag = _stamp()
    if own_audio:
        audio = own_audio
    else:
        if not script or not script.strip():
            raise gr.Error("Type the script, or upload your own voice.")
        audio = make_voice(script, voice, out_wav=f"voice_{tag}.wav", speed=speed)

    secs = audio_seconds(audio)
    video_secs = min(10, max(3, math.ceil(secs)))
    prompt = motion_prompt.strip()
    if add_talking:
        prompt += ", the person is talking to the camera, natural lip and face movement"

    moving = generate(image, prompt, output=f"moving_{tag}.mp4", seconds=video_secs)
    final = make_lipsync_video(moving, audio, out_path=f"cinematic_{tag}.mp4", sharpen=sharpen)
    note = f"✅ Done! Voice {secs:.1f}s."
    if secs > 10:
        note += " (Voice is longer than 10s, so the video loops to fit.)"
    return audio, final, note


# ---------- Tab 2: moving video + narrator ----------
def voiceover(image, motion_prompt, script, voice, speed, seconds):
    from generate import generate
    from talking import add_voiceover

    if image is None:
        raise gr.Error("Please upload a starting image.")
    if not motion_prompt or not motion_prompt.strip():
        raise gr.Error("Describe the motion (what should move in the video).")
    if not script or not script.strip():
        raise gr.Error("Type the narrator script.")

    tag = _stamp()
    audio = make_voice(script, voice, out_wav=f"voice_{tag}.wav", speed=speed)
    secs = audio_seconds(audio)
    note = ""
    if secs > seconds:
        note = (f"⚠️ Voice is {secs:.1f}s but video is {seconds}s – the end of the voice is cut. "
                "Shorten the script or speed up the voice.")

    silent = generate(image, motion_prompt, output=f"silent_{tag}.mp4", seconds=seconds)
    final = add_voiceover(silent, audio, out_path=f"voiceover_{tag}.mp4")
    return final, note or f"✅ Done! Voice length: {secs:.1f}s."


# ---------- Tab 3: silent video ----------
def silent(image, prompt, seconds, fps, steps, max_side, seed):
    from generate import generate

    if image is None:
        raise gr.Error("Please upload an image first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please type a prompt describing the motion.")
    return generate(image, prompt, output=f"video_{_stamp()}.mp4", seconds=seconds,
                    fps=int(fps), steps=int(steps), max_side=int(max_side), seed=int(seed))


def voice_preview(script, voice, speed):
    if not script or not script.strip():
        raise gr.Error("Type a script first.")
    wav = make_voice(script, voice, out_wav=f"preview_{_stamp()}.wav", speed=speed)
    return wav, f"Voice length: {audio_seconds(wav):.1f} seconds"


SCRIPT_TIP = ("Tip: about 25 words ≈ 10 seconds. "
              "Halimbawa: \"Magandang araw! Ako si Ana. Subukan mo ang aming bagong produkto ngayon!\"")

with gr.Blocks(title="Free AI Video Studio") as demo:
    gr.Markdown("# 🎬 Free AI Video Studio\nEnglish 🇺🇸 + Tagalog 🇵🇭 voices · 100% free & open-source")

    with gr.Tab("🗣️ Talking character"):
        gr.Markdown("Upload a picture with **one clear, front-facing face**, type what they say, and pick a voice.")
        with gr.Row():
            with gr.Column():
                t_img = gr.Image(type="filepath", label="Character picture")
                t_script = gr.Textbox(label="Script (English or Tagalog)", lines=4, info=SCRIPT_TIP)
                t_voice = gr.Dropdown(VOICE_NAMES, value=DEFAULT_VOICE, label="Voice")
                t_speed = gr.Slider(-30, 30, value=0, step=5, label="Voice speed (%)")
                t_prev_btn = gr.Button("🔊 Preview voice")
                with gr.Accordion("Use my own recording instead (optional)", open=False):
                    t_own = gr.Audio(type="filepath", label="Upload or record your voice")
                with gr.Accordion("Settings", open=False):
                    t_full = gr.Checkbox(True, label="Keep the whole picture (off = zoom on face)")
                    t_enh = gr.Checkbox(True, label="Sharpen face (slower, looks better)")
                    t_size = gr.Radio([256, 512], value=256, label="Face quality (512 = sharper, slower)")
                    t_expr = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Expression strength")
                t_btn = gr.Button("Make talking video", variant="primary")
            with gr.Column():
                t_audio = gr.Audio(label="Voice")
                t_video = gr.Video(label="Talking video")
                t_status = gr.Markdown()
        t_prev_btn.click(voice_preview, [t_script, t_voice, t_speed], [t_audio, t_status])
        t_btn.click(talking, [t_img, t_script, t_voice, t_speed, t_own, t_full, t_enh, t_size, t_expr],
                    [t_audio, t_video, t_status])

    with gr.Tab("🎥 Cinematic talking"):
        gr.Markdown("**Moving body + camera like a movie, AND she talks with lip-sync.** "
                    "Slowest option (about 10–20 min per video on the free GPU). "
                    "Use a picture where the face is clear and not too small.")
        with gr.Row():
            with gr.Column():
                c_img = gr.Image(type="filepath", label="Starting picture")
                c_motion = gr.Textbox(
                    label="Motion prompt (English works best)", lines=3,
                    placeholder="A Filipino woman in a floral dress stands in a sari-sari store, smiling and gently "
                                "swaying, holding up a phone, soft natural light, camera slowly zooms in, cinematic")
                c_script = gr.Textbox(label="Script (English or Tagalog)", lines=4, info=SCRIPT_TIP)
                c_voice = gr.Dropdown(VOICE_NAMES, value=DEFAULT_VOICE, label="Voice")
                c_speed = gr.Slider(-30, 30, value=0, step=5, label="Voice speed (%)")
                with gr.Accordion("Use my own recording instead (optional)", open=False):
                    c_own = gr.Audio(type="filepath", label="Upload or record your voice")
                with gr.Accordion("Settings", open=False):
                    c_talk = gr.Checkbox(True, label="Tell the video model she is talking (recommended)")
                    c_sharp = gr.Checkbox(True, label="Sharpen face (slower, looks better)")
                c_btn = gr.Button("Make cinematic talking video", variant="primary")
            with gr.Column():
                c_audio = gr.Audio(label="Voice")
                c_video = gr.Video(label="Cinematic video")
                c_status = gr.Markdown()
        c_btn.click(cinematic, [c_img, c_motion, c_script, c_voice, c_speed, c_own, c_talk, c_sharp],
                    [c_audio, c_video, c_status])

    with gr.Tab("🎙️ Video + voice-over"):
        gr.Markdown("The picture comes to life (moving scene) and a narrator reads your script. "
                    "No lip-sync – best for product promos, scenery, storytelling.")
        with gr.Row():
            with gr.Column():
                v_img = gr.Image(type="filepath", label="Starting image")
                v_motion = gr.Textbox(label="Motion prompt (English works best)", lines=2,
                                      placeholder="Slow zoom in on the product, soft light, leaves gently moving")
                v_script = gr.Textbox(label="Narrator script (English or Tagalog)", lines=4, info=SCRIPT_TIP)
                v_voice = gr.Dropdown(VOICE_NAMES, value=DEFAULT_VOICE, label="Voice")
                v_speed = gr.Slider(-30, 30, value=0, step=5, label="Voice speed (%)")
                v_secs = gr.Slider(4, 10, value=10, step=1, label="Video length (seconds)")
                v_btn = gr.Button("Make video with voice-over", variant="primary")
            with gr.Column():
                v_video = gr.Video(label="Result")
                v_status = gr.Markdown()
        v_btn.click(voiceover, [v_img, v_motion, v_script, v_voice, v_speed, v_secs], [v_video, v_status])

    with gr.Tab("🎬 Silent video"):
        with gr.Row():
            with gr.Column():
                s_img = gr.Image(type="filepath", label="Starting image")
                s_prompt = gr.Textbox(label="Prompt", lines=3,
                                      placeholder="A golden retriever runs across the beach, camera slowly follows")
                with gr.Accordion("Settings", open=False):
                    s_secs = gr.Slider(2, 10, value=10, step=1, label="Length (seconds)")
                    s_fps = gr.Slider(8, 24, value=16, step=1, label="FPS")
                    s_steps = gr.Slider(10, 50, value=30, step=1, label="Quality steps (more = slower)")
                    s_side = gr.Dropdown([384, 512, 640, 768], value=512, label="Max resolution (long side)")
                    s_seed = gr.Number(value=42, label="Seed (change for a different result)")
                s_btn = gr.Button("Generate video", variant="primary")
            with gr.Column():
                s_video = gr.Video(label="Result")
        s_btn.click(silent, [s_img, s_prompt, s_secs, s_fps, s_steps, s_side, s_seed], s_video)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
