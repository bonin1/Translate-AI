import torch
from transformers import MarianMTModel, MarianTokenizer, MarianConfig
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import pygame
from pathlib import Path
import json
import keyboard


class TranslationChat:
    def __init__(self, model_path="saved_models/best_model_epoch_1"):
        print("Initializing translator...")
        self.model_path = Path(model_path)
        self.current_direction = "en2de"

        try:
            self.en_de_model_name = "Helsinki-NLP/opus-mt-en-de"
            self.de_en_model_name = "Helsinki-NLP/opus-mt-de-en"
            print(f"Loading base configurations...")

            config = MarianConfig.from_pretrained(self.en_de_model_name)
            config.d_model = 512
            config.encoder_attention_heads = 8
            config.encoder_ffn_dim = 2048
            config.decoder_attention_heads = 8
            config.decoder_ffn_dim = 2048

            print("Loading models and tokenizers...")
            self.en_de_tokenizer = MarianTokenizer.from_pretrained(self.en_de_model_name)
            self.de_en_tokenizer = MarianTokenizer.from_pretrained(self.de_en_model_name)

            self.en_de_model = MarianMTModel.from_pretrained(
                self.model_path,
                config=config,
                ignore_mismatched_sizes=True
            )
            self.de_en_model = MarianMTModel.from_pretrained(self.de_en_model_name)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            self.en_de_model.to(self.device)
            self.de_en_model.to(self.device)
            self.en_de_model.eval()
            self.de_en_model.eval()

            print("Initializing speech recognition...")
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            print("Initializing audio...")
            pygame.mixer.init()

            print("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Setup complete! Ready to translate.")

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def translate(self, text, direction="en2de"):
        """Translate text between English and German"""
        try:
            if direction == "en2de":
                model = self.en_de_model
                tokenizer = self.en_de_tokenizer
            else:
                model = self.de_en_model
                tokenizer = self.de_en_tokenizer

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )

            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = translation.replace('▁', ' ').strip()
            translation = ' '.join(translation.split())
            return translation
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return ""

    def speak_text(self, text, lang):
        """Convert text to speech"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name

            tts = gTTS(text=text, lang=lang)
            tts.save(temp_filename)

            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")

    def listen_for_speech(self, language="en-US"):
        """Listen for speech input"""
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio, language=language)
                return text
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Could not request results; {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None

    def chat(self):
        """Main chat loop"""
        print("\n🌟 Welcome to the Bidirectional Translator!")
        print("📌 Instructions:")
        print("- Press Enter to start speaking")
        print("- Type 'switch' to change translation direction")
        print("- Type 'quit' to exit")

        while True:
            try:
                direction_text = "English → German" if self.current_direction == "en2de" else "German → English"
                print(f"\n🔄 Current mode: {direction_text}")
                command = input("Press Enter to speak, type 'switch' to change direction, or 'quit' to exit: ").lower()

                if command == 'quit':
                    print("👋 Goodbye!")
                    break
                elif command == 'switch':
                    self.current_direction = "de2en" if self.current_direction == "en2de" else "en2de"
                    continue

                source_lang = "en-US" if self.current_direction == "en2de" else "de-DE"
                target_lang = "de" if self.current_direction == "en2de" else "en"

                text = self.listen_for_speech(source_lang)
                if not text:
                    continue

                print(f"📝 You said: {text}")

                translation = self.translate(text, self.current_direction)
                print(f"🔄 Translation: {translation}")

                self.speak_text(translation, target_lang)

            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    try:
        try:
            import sacremoses
        except ImportError:
            print("Installing sacremoses...")
            import pip
            pip.main(['install', 'sacremoses'])

        chat = TranslationChat()
        chat.chat()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("\nPlease ensure you have all required packages installed:")
        print(
            "pip install torch transformers pandas tqdm nltk gTTS pygame SpeechRecognition pyaudio sacremoses keyboard")


if __name__ == "__main__":
    main()