import pytest
import os
from src.config.settings import settings

def test_settings_paths():
    """Verify that settings resolve paths properly"""
    assert "data" in settings.DATA_DIR
    assert "raw" in settings.RAW_DATA_PATH
    assert isinstance(settings.PROJECT_NAME, str)
