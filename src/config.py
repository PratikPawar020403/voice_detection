import os

# Base Paths
# This file is in src/, so project root is one level up
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Ensure data root exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

RAW_HUMAN_DIR = os.path.join(DATA_DIR, 'raw', 'human')
RAW_AI_DIR = os.path.join(DATA_DIR, 'raw', 'ai')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Languages
LANGUAGES = {
    'en': 'English',
    'ta': 'Tamil',
    'hi': 'Hindi',
    'ml': 'Malayalam',
    'te': 'Telugu'
}

# Audio Settings
SAMPLE_RATE = 16000
DURATION_LIMIT = 10  # Seconds to trim/pad to
