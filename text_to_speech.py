import boto3
import os
import subprocess

# Vicki, Daniel, Lennart, Tina
def text_to_speech(text, output_file="speech.mp3", voice_id="Lennart", engine="generative"):
    polly_client = boto3.client('polly')

    response = polly_client.synthesize_speech(
        VoiceId=voice_id,
        OutputFormat="mp3",
        Text=text,
        Engine=engine,
    )

    output_path = os.path.abspath(output_file)

    # Speichern der Audiodatei
    if "AudioStream" in response:
        with open(output_path, "wb") as file:
            file.write(response["AudioStream"].read())

    # Audiodatei abspielen
    try:
        subprocess.run(
            ["mpv", output_path],
            check=False,
        )
    except FileNotFoundError:
        print("mpv wurde nicht gefunden.")


if __name__ == "__main__":
    text = "Hallo, das ist ein Beispiel für Text-to-Speech mit Amazon Polly."
    text_to_speech(text)