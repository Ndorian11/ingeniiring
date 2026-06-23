import pytest
from unittest.mock import MagicMock, patch
from object_detection import ObjectDetector


@pytest.fixture
def detector():
    # Мокаем инициализацию модели YOLO, чтобы она не скачивалась при создании объекта
    with patch('object_detection.YOLO') as mock_yolo:
        detector_instance = ObjectDetector()
        yield detector_instance


def test_translation_dict_completeness(detector):
    """Проверка, что ключевые слова присутствуют в словаре перевода"""
    from object_detection import TRANSLATION_DICT
    assert 'person' in TRANSLATION_DICT
    assert TRANSLATION_DICT['person'] == 'человек'
    assert 'cat' in TRANSLATION_DICT


@pytest.mark.parametrize("input_objects, lang, expected", [
    (['person'], 'ru', "На этом изображении человек"),
    (['person'], 'en', "На этом изображении person"),
    (['cat', 'dog'], 'ru', "На этом изображении кот и собака"),
    (['cat', 'cat', 'dog'], 'ru', "На этом изображении 2 кот и собака"),
    (['person', 'cat', 'dog'], 'ru', "На этом изображении человек, кот и собака"),
    ([], 'ru', "На изображении не обнаружено объектов")
])
def test_format_text_for_speech(detector, input_objects, lang, expected):
    """Тестирование корректности формирования фраз для озвучки на разных языках"""
    result = detector.format_text_for_speech(input_objects, language=lang)
    assert result == expected


@patch('object_detection.gTTS')
def test_generate_speech(mock_gtts, detector):
    """Тестирование вызова генерации речи без реального создания файла"""
    mock_tts_instance = MagicMock()
    mock_gtts.return_value = mock_tts_instance

    text = "Проверка звука"
    output_file = "test_audio.mp3"

    result = detector.generate_speech(text, language='ru', output_file=output_file)

    # Проверяем, что gTTS был вызван с правильными параметрами
    mock_gtts.assert_called_once_with(text=text, lang='ru', slow=False)
    mock_tts_instance.save.assert_called_once_with(output_file)
    assert result == output_file