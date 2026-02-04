import gradio as gr
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.api.inference import predict_pipeline
from src.api.lid import identify_language

def file_to_bytes(file):
    with open(file, "rb") as f:
        return f.read()

def analyze_audio_formatted(audio_file):
    if audio_file is None:
        return None, "No file.", "Unknown"
    
    try:
        # 1. Voice Detection (AI vs Human)
        audio_bytes = file_to_bytes(audio_file)
        result = predict_pipeline(audio_bytes)
        
        # Label format for Gradio
        if result['result'] == "AI_GENERATED":
            scores = {"AI_GENERATED": result['confidence'], "HUMAN": 1 - result['confidence']}
        else:
            scores = {"HUMAN": result['confidence'], "AI_GENERATED": 1 - result['confidence']}
            
        # 2. Language ID
        lang_id = identify_language(audio_file)
            
        return scores, result['explanation'], lang_id
        
    except Exception as e:
        return None, str(e), "Error"

# Custom CSS for a professional look
custom_css = """
.container {max_width: 800px; margin: auto; padding-top: 20px}
.header {text-align: center; color: #333}
.result-box {font-size: 1.5em; font-weight: bold; text-align: center}
"""

with gr.Blocks(css=custom_css, title="AI Voice Detector") as demo:
    gr.Markdown("# 🕵️ AI Voice Detection System")
    gr.Markdown("Upload an audio file (MP3/WAV/FLAC) to check if it's Human or AI-generated and identify the language.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            submit_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column():
            result_label = gr.Label(label="Prediction")
            lang_label = gr.Textbox(label="Detected Language")
            explanation_box = gr.Textbox(label="Explanation", lines=3)

    submit_btn.click(
        fn=analyze_audio_formatted,
        inputs=[audio_input],
        outputs=[result_label, explanation_box, lang_label]
    )

if __name__ == "__main__":
    demo.launch(share=True)
