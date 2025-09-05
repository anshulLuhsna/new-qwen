# /workspace/voice_agent.py

import torch
import soundfile as sf
import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class QwenVoiceAgent:
    """
    A voice agent powered by the Qwen 2.5 Omni 7B model, capable of
    speech-in and speech-out interactions.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B", speaker="Chelsie"):
        """
        Initializes the agent by loading the model and processor.

        Args:
            model_name (str): The Hugging Face model identifier.
            speaker (str): The default voice for audio output ('Chelsie' or 'Ethan').
        """
        print(f"Loading {model_name}. This may take several minutes...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.speaker = speaker

        # Load the model with optimizations for A100/H100
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        # The Talker (audio output module) is enabled by default.
        # To disable it and save ~2GB VRAM for text-only tasks, you could call:
        # self.model.disable_talker()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        # This system prompt is MANDATORY for enabling audio output
        self.system_prompt = {
            "role": "system",
            "content":
        }
        print("Qwen 2.5 Omni model loaded successfully.")

    def _generate_response(self, conversation):
        """
        Internal helper method to process a conversation and generate a response.
        """
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation)

        inputs = self.processor(
            text=text_prompt, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True
        ).to(self.model.device)

        # Generate both text and audio
        text_ids, audio = self.model.generate(
            **inputs,
            speaker=self.speaker,
            return_audio=True # Explicitly request audio output
        )

        response_text = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response_text, audio.cpu().numpy()

    def speak_from_text(self, user_text):
        """
        Generates a voice reply from a text input.

        Args:
            user_text (str): The text input from the user.

        Returns:
            tuple: A tuple containing the response text (str) and audio (numpy array).
        """
        conversation = [
            self.system_prompt,
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            }
        ]
        return self._generate_response(conversation)

    def speak_from_audio(self, user_audio_file):
        """
        Generates a voice reply from an audio file input.

        Args:
            user_audio_file (str): Path to the user's audio file.

        Returns:
            tuple: A tuple containing the response text (str) and audio (numpy array).
        """
        conversation = [
            self.system_prompt,
            {
                "role": "user",
                "content": [{"type": "audio", "audio": user_audio_file}]
            }
        ]
        return self._generate_response(conversation)

if __name__ == "__main__":
    # Example usage for interactive command-line testing
    agent = QwenVoiceAgent()
    
    print("\n--- Qwen 2.5 Omni Voice Agent ---")
    print("Enter text to get a spoken response.")
    print("To use an audio file, type: audio:/path/to/your/audio.wav")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        try:
            if user_input.lower().startswith("audio:"):
                file_path = user_input[6:].strip()
                if not os.path.isfile(file_path):
                    print("System: Audio file not found. Please provide a valid path.")
                    continue
                print("System: Processing audio input...")
                reply, audio_out = agent.speak_from_audio(file_path)
            else:
                print("System: Processing text input...")
                reply, audio_out = agent.speak_from_text(user_input)

            print(f"Agent: {reply}")

            # Save the agent's spoken response for verification
            output_filename = "agent_reply.wav"
            sf.write(output_filename, audio_out.reshape(-1), samplerate=24000)
            print(f"System: Voice reply saved to {output_filename}\n")

        except Exception as e:
            print(f"An error occurred: {e}")
