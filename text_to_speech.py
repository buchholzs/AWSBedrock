import boto3
import os
import re
import subprocess

# Vicki, Daniel, Lennart, Tina


def split_text_for_polly(text, max_chars=2500):
    """Teilt Text in sichere Abschnitte für Polly auf."""
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    parts = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            parts.append(current)
            current = sentence

    if current:
        parts.append(current)

    chunks = []
    for part in parts:
        while len(part) > max_chars:
            split_index = part.rfind(" ", 0, max_chars)
            if split_index <= 0:
                split_index = max_chars
            chunks.append(part[:split_index].strip())
            part = part[split_index:].strip()
        if part:
            chunks.append(part)

    return chunks


def stream_audio_to_mpv(audio_stream):
    """Spielt einen Polly-AudioStream direkt über mpv ab, ohne ihn erst lokal zu speichern."""
    try:
        process = subprocess.Popen(
            ["mpv", "--no-video", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("mpv wurde nicht gefunden.")
        return None

    try:
        for chunk in audio_stream.iter_chunks(chunk_size=4096):
            if chunk:
                process.stdin.write(chunk)
                process.stdin.flush()
        process.stdin.close()
        process.wait()
    except Exception:
        if process.stdin:
            process.stdin.close()
        process.kill()
        raise

    return process.returncode


def write_audio_to_file(audio_stream, output_file):
    output_path = os.path.abspath(output_file)
    with open(output_path, "wb") as file:
        for chunk in audio_stream.iter_chunks(chunk_size=4096):
            if chunk:
                file.write(chunk)
    return output_path


def text_to_speech(text, output_file="speech.mp3", voice_id="Lennart", engine="generative", max_chars=2500, save_to_file=False):
    polly_client = boto3.client('polly')
    chunks = split_text_for_polly(text, max_chars=max_chars)

    if not chunks:
        return None

    current_output_paths = []
    output_path = os.path.abspath(output_file) if output_file else None

    try:
        process = subprocess.Popen(
            ["mpv", "--no-video", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("mpv wurde nicht gefunden.")
        return None

    if save_to_file and output_path:
        file_handle = open(output_path, "wb")

    for chunk in chunks:
        response = polly_client.synthesize_speech(
            VoiceId=voice_id,
            OutputFormat="mp3",
            Text=chunk,
            Engine=engine,
        )

        audio_stream = response.get("AudioStream")
        if not audio_stream:
            continue

        if save_to_file and output_path:
            for data in audio_stream.iter_chunks(chunk_size=4096):
                if data:
                    file_handle.write(data)
            current_output_paths.append(output_path)
        else:
            for data in audio_stream.iter_chunks(chunk_size=4096):
                if data:
                    process.stdin.write(data)
                    process.stdin.flush()

    if save_to_file and output_path:
        file_handle.close()

    if process.stdin:
        process.stdin.close()
    process.wait()

    if save_to_file:
        return current_output_paths[-1] if current_output_paths else None
    return None


if __name__ == "__main__":
    text = "Hallo, das ist ein Beispiel für Text-to-Speech mit Amazon Polly. " * 3
    text_to_speech(text)