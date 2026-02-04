import os
import asyncio
import pandas as pd
import soundfile as sf
import librosa
from gtts import gTTS
import edge_tts
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import LANGUAGES, RAW_AI_DIR, SAMPLE_RATE

# Text prompts for generation (Mix of lengths and types)
# We will generate permutations or use a larger corpus if needed. 
# For now, a small set repeated with different voices/speeds is a good start.
TEXT_CORPUS = {
    'en': [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, this is an AI voice generation test.",
        "Artificial intelligence is transforming the world.",
        "Can you tell me the time please?",
        "I am not a human, but I sound like one.",
        "Verification code is one two three four.",
        "Open the pod bay doors, HAL.",
        "The weather today is sunny with a chance of rain.",
        "Please confirm your identity.",
        "This is a secure channel."
    ],
    'ta': [
        "வணக்கம், எப்படி இருக்கிறீர்கள்?",  # Hello, how are you?
        "இது ஒரு செயற்கை நுண்ணறிவு குரல் சோதனை.", # This is an AI voice test.
        "தமிழ் உலகின் மூத்த மொழிகளில் ஒன்று.", # Tamil is one of the oldest languages.
        "இன்று வானிலை மிக நன்றாக உள்ளது.", # The weather is very good today.
        "தயவுசெய்து உங்கள் அடையாளத்தை உறுதிப்படுத்தவும்." # Please verify your identity.
    ],
    'hi': [
        "नमस्ते, आप कैसे हैं?", # Hello, how are you?
        "यह एक एआई आवाज़ परीक्षण है।", # This is an AI voice test.
        "भारत एक विशाल देश है।", # India is a huge country.
        "कृपया अपना पासवर्ड दर्ज करें।", # Please enter your password.
        "मौसम आज बहुत सुहावना है।" # The weather is very pleasant today.
    ],
    'ml': [
        "നമസ്കാരം, സുഖമാണോ?", # Hello, are you fine?
        "ഇതൊരു നിർമ്മിത ബുദ്ധി പരീക്ഷണമാണ്.", # This is an AI test.
        "കേരളം ദൈവത്തിന്റെ സ്വന്തം നാടാണ്.", # Kerala is God's own country.
        "ദയവായി വാതിൽ തുറക്കൂ.", # Please open the door.
        "ഇന്നത്തെ കാലാവസ്ഥ എങ്ങനെയുണ്ട്?" # How is today's weather?
    ],
    'te': [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?", # Hello, how are you?
        "ఇది ఒక కృత్రిమ మేధస్సు పరీక్ష.", # This is an AI test.
        "తెలుగు చాలా తీయని భాష.", # Telugu is a very sweet language.
        "దయచేసి మీ పేరు చెప్పండి.", # Please tell your name.
        "ఈ రోజు వర్షం పడే అవకాశం ఉంది." # There is a chance of rain today.
    ]
}

async def generate_edge_tts(text, voice, output_path):
    """Generate audio using Edge TTS"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_gtts(text, lang_code, output_path):
    """Generate audio using Google TTS"""
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tts.save(output_path)

async def main():
    if not os.path.exists(RAW_AI_DIR):
        os.makedirs(RAW_AI_DIR)

    data_records = []
    
    # Edge TTS Voices Map (Approximate)
    EDGE_VOICES = {
        'en': ['en-US-ChristopherNeural', 'en-US-JennyNeural', 'en-GB-SoniaNeural'],
        'ta': ['ta-IN-ValluvarNeural', 'ta-IN-PallaviNeural'],
        'hi': ['hi-IN-MadhurNeural', 'hi-IN-SwaraNeural'],
        'ml': ['ml-IN-MidhunNeural', 'ml-IN-SobhanaNeural'],
        'te': ['te-IN-MohanNeural', 'te-IN-ShrutiNeural']
    }

    target_per_lang = 50
    
    for lang_code, lang_name in LANGUAGES.items():
        print(f"Generating AI samples for {lang_name} ({lang_code})...")
        lang_dir = os.path.join(RAW_AI_DIR, lang_code)
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)
            
        texts = TEXT_CORPUS.get(lang_code, TEXT_CORPUS['en']) # Fallback to English if missing
        count = 0
        
        # 1. Edge TTS Generation
        voices = EDGE_VOICES.get(lang_code, [])
        for voice in voices:
            for text in texts:
                if count >= target_per_lang // 2: # Do half with Edge, half with gTTS
                    break
                
                fname = f"ai_edge_{lang_code}_{count:04d}.mp3"
                fpath = os.path.join(lang_dir, fname)
                
                try:
                    await generate_edge_tts(text, voice, fpath)
                    
                    # Verify and convert to consistent format if needed (deferred to preprocessing)
                    # For now just save record
                    data_records.append({
                        'filename': fname,
                        'language': lang_code,
                        'path': fpath,
                        'source': 'edge_tts',
                        'voice_engine': voice
                    })
                    count += 1
                except Exception as e:
                    print(f"Error generating Edge TTS for {lang_code}: {e}")

        # 2. gTTS Generation (Fill the rest)
        gtts_lang = lang_code
        # gTTS mappings usually match ISO codes, but check docs if failures occur.
        # ta, hi, ml, te are supported.
        
        for text in texts:
            if count >= target_per_lang:
                break
                
            fname = f"ai_gtts_{lang_code}_{count:04d}.mp3"
            fpath = os.path.join(lang_dir, fname)
            
            try:
                generate_gtts(text, gtts_lang, fpath)
                
                data_records.append({
                    'filename': fname,
                    'language': lang_code,
                    'path': fpath,
                    'source': 'gtts',
                    'voice_engine': 'gtts_standard'
                })
                count += 1
            except Exception as e:
                print(f"Error generating gTTS for {lang_code}: {e}")
                
    # Save Metadata
    df = pd.DataFrame(data_records)
    csv_path = os.path.join(RAW_AI_DIR, 'ai_samples.csv')
    df.to_csv(csv_path, index=False)
    print(f"AI Data Generation Complete! Saved to {csv_path}")

if __name__ == "__main__":
    asyncio.run(main())
