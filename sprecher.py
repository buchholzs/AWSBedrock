import boto3

polly = boto3.client("polly", region_name="eu-west-1")

voices = polly.describe_voices(LanguageCode="fr-FR")["Voices"]

for v in voices:
    engines = v.get("SupportedEngines", [])
    print(v["Id"], engines)