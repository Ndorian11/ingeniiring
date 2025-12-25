"""
Пример использования модуля распознавания объектов
"""

from object_detection import ObjectDetector

def main():
    print("=" * 70)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДУЛЯ РАСПОЗНАВАНИЯ ОБЪЕКТОВ")
    print("=" * 70)
    
    # Инициализация детектора
    print("\n[1/3] Инициализация детектора...")
    detector = ObjectDetector('yolov8n.pt')
    
    # Пример 1: Обработка изображения по URL
    print("\n[2/3] Обработка тестового изображения...")
    test_url = 'https://ultralytics.com/images/bus.jpg'
    
    result = detector.process_image(
        test_url,
        confidence=0.4,
        language='ru',
        generate_audio=True
    )
    
    if result:
        print("\n[3/3] Результаты:")
        print("-" * 70)
        print(f"Найдено объектов: {result['total_objects']}")
        print(f"Текст для озвучивания: {result['text']}")
        
        if result['audio_file']:
            print(f"Аудио файл создан: {result['audio_file']}")
        
        print("\nДетали распознавания:")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['name_ru']} ({det['name']}) - уверенность: {det['confidence']:.2%}")
        
        print("-" * 70)
        print("\n✓ Обработка завершена успешно!")
    else:
        print("\n✗ Ошибка обработки изображения")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

