import boto3
from botocore.exceptions import BotoCoreError, ClientError

from text_to_speech import text_to_speech

# 1. Bedrock-Runtime Client initialisieren
# (Achte darauf, eine Region zu wählen, in der Nova Pro aktiv ist, z. B. 'us-east-1')
# us-east-1, eu-west-1, eu-central-1
bedrock_client = boto3.client('bedrock-runtime', region_name='eu-west-1')

# Exakte Model-ID für Amazon Nova Pro
# amazon.nova-pro-v1:0 für us-east-1
# eu.amazon.nova-pro-v1:0 für eu-west-1 und eu-central-1
MODEL_ID = 'eu.amazon.nova-pro-v1:0'

# AWS Translate-Client für die Übersetzung der Antwort ins Französische
translate_client = boto3.client('translate', region_name='eu-west-1')


def translate_to_french(text):
  if not text or not text.strip():
    return text

  response = translate_client.translate_text(
      Text=text,
      SourceLanguageCode='auto',
      TargetLanguageCode='fr',
  )

  return response.get('TranslatedText', text)


def start_chatbot():
  print(f'--- Chatbot mit Amazon Nova Pro ({MODEL_ID}) ---')
  print("Tippe 'exit' oder 'quit', um das Gespräch zu beenden.\n")

  # Liste zur Speicherung des Chatverlaufs (Kontext)
  messages = []

  while True:
    try:
      # Benutzereingabe einlesen
      user_input = input('Du: ')
      if user_input.lower() in ['exit', 'quit']:
        print('Chat beendet. Auf Wiedersehen!')
        break

      if not user_input.strip():
        continue

      # Nachricht im von der Converse API erwarteten Format anhängen
      messages.append({'role': 'user', 'content': [{'text': user_input}]})

      # 2. API-Aufruf an Amazon Bedrock
      response = bedrock_client.converse(
          modelId=MODEL_ID,
          messages=messages,
          inferenceConfig={
              'maxTokens': 1024,
              'temperature': 0.7,
          },
      )

      # 3. Antwort des Modells auslesen
      response_message = response['output']['message']
      bot_reply = response_message['content'][0]['text']
      print(f'\nNova Pro: {bot_reply}\n')

      french_reply = translate_to_french(bot_reply)
      print(f'\nNova Pro (französisch): {french_reply}\n')

      # Übersetzte Antwort mit Amazon Polly ausgeben
      text_to_speech(bot_reply, voice_id='Lennart')
      text_to_speech(french_reply, voice_id='Remi', engine='neural')

      # Die Antwort des Assistenten dem Verlauf hinzufügen, 
      # damit das Modell sich im nächsten Schritt an den Kontext erinnert
      messages.append(response_message)

    except (BotoCoreError, ClientError) as error:
      print(f'\n[AWS-Fehler aufgetreten]: {error}\n')
    except KeyboardInterrupt:
      print('\nChat durch Benutzer abgebrochen.')
      break


if __name__ == '__main__':
  start_chatbot()