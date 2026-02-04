"""
Pre-deployment edge case testing script.
Tests all error handling paths for competition compliance.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

API_KEY = "voicedetect_2024_secure_key"
VALID_HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def test_health_check():
    """Test health endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    print("✓ Health check passed")

def test_invalid_api_key():
    """Test invalid API key returns 403 with error JSON"""
    response = client.post(
        "/api/voice-detection",
        headers={"x-api-key": "wrong_key", "Content-Type": "application/json"},
        json={"language": "English", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    assert response.status_code == 403
    data = response.json()
    assert data["status"] == "error"
    assert "Invalid API key" in data["message"]
    print("✓ Invalid API key test passed")

def test_missing_api_key():
    """Test missing API key returns 403"""
    response = client.post(
        "/api/voice-detection",
        headers={"Content-Type": "application/json"},
        json={"language": "English", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    assert response.status_code == 403
    data = response.json()
    assert data["status"] == "error"
    print("✓ Missing API key test passed")

def test_unsupported_language_code():
    """Test language code 'en' is rejected (only full names allowed)"""
    response = client.post(
        "/api/voice-detection",
        headers=VALID_HEADERS,
        json={"language": "en", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Unsupported language" in data["message"]
    print("✓ Language code 'en' rejected (correct)")

def test_unsupported_language_spanish():
    """Test unsupported language 'Spanish' is rejected"""
    response = client.post(
        "/api/voice-detection",
        headers=VALID_HEADERS,
        json={"language": "Spanish", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Spanish" in data["message"]
    print("✓ Unsupported language 'Spanish' rejected")

def test_wrong_audio_format():
    """Test non-mp3 format is rejected"""
    response = client.post(
        "/api/voice-detection",
        headers=VALID_HEADERS,
        json={"language": "English", "audioFormat": "wav", "audioBase64": "dGVzdA=="}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "mp3" in data["message"].lower()
    print("✓ Wrong audio format test passed")

def test_invalid_base64():
    """Test invalid Base64 returns proper error"""
    response = client.post(
        "/api/voice-detection",
        headers=VALID_HEADERS,
        json={"language": "English", "audioFormat": "mp3", "audioBase64": "!!!invalid!!!"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Base64" in data["message"]
    print("✓ Invalid Base64 test passed")

def test_valid_language_names():
    """Test all valid language names are accepted (schema validation only)"""
    valid_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    for lang in valid_languages:
        # Note: This will fail on audio processing but should pass language validation
        response = client.post(
            "/api/voice-detection",
            headers=VALID_HEADERS,
            json={"language": lang, "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
        )
        # Should NOT be 400 for language error
        if response.status_code == 400:
            data = response.json()
            assert "Unsupported language" not in data.get("message", ""), f"Language {lang} was rejected!"
    print("✓ All valid language names accepted")

def test_case_insensitive_language():
    """Test language names are case-insensitive"""
    for lang in ["ENGLISH", "english", "English", "TAMIL", "hindi"]:
        response = client.post(
            "/api/voice-detection",
            headers=VALID_HEADERS,
            json={"language": lang, "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
        )
        if response.status_code == 400:
            data = response.json()
            assert "Unsupported language" not in data.get("message", ""), f"Language {lang} was rejected!"
    print("✓ Case-insensitive language names work")

if __name__ == "__main__":
    print("=" * 50)
    print("PRE-DEPLOYMENT EDGE CASE TESTS")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_invalid_api_key,
        test_missing_api_key,
        test_unsupported_language_code,
        test_unsupported_language_spanish,
        test_wrong_audio_format,
        test_invalid_base64,
        test_valid_language_names,
        test_case_insensitive_language,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test.__name__} - {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL EDGE CASE TESTS PASSED!")
    else:
        print(f"⚠️ {failed} tests failed")
    print("=" * 50)
