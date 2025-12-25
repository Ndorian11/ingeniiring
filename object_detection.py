"""
Модуль для распознавания объектов на изображениях с озвучиванием результатов
"""

import torch
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import requests
from io import BytesIO


# Словарь для перевода названий классов с английского на русский
TRANSLATION_DICT = {
    'person': 'человек',
    'bicycle': 'велосипед',
    'car': 'машина',
    'motorcycle': 'мотоцикл',
    'airplane': 'самолет',
    'bus': 'автобус',
    'train': 'поезд',
    'truck': 'грузовик',
    'boat': 'лодка',
    'traffic light': 'светофор',
    'fire hydrant': 'пожарный гидрант',
    'stop sign': 'знак стоп',
    'parking meter': 'парковочный счетчик',
    'bench': 'скамейка',
    'bird': 'птица',
    'cat': 'кот',
    'dog': 'собака',
    'horse': 'лошадь',
    'sheep': 'овца',
    'cow': 'корова',
    'elephant': 'слон',
    'bear': 'медведь',
    'zebra': 'зебра',
    'giraffe': 'жираф',
    'backpack': 'рюкзак',
    'umbrella': 'зонт',
    'handbag': 'сумка',
    'tie': 'галстук',
    'suitcase': 'чемодан',
    'frisbee': 'фрисби',
    'skis': 'лыжи',
    'snowboard': 'сноуборд',
    'sports ball': 'мяч',
    'kite': 'воздушный змей',
    'baseball bat': 'бейсбольная бита',
    'baseball glove': 'бейсбольная перчатка',
    'skateboard': 'скейтборд',
    'surfboard': 'доска для серфинга',
    'tennis racket': 'теннисная ракетка',
    'bottle': 'бутылка',
    'wine glass': 'бокал',
    'cup': 'чашка',
    'fork': 'вилка',
    'knife': 'нож',
    'spoon': 'ложка',
    'bowl': 'миска',
    'banana': 'банан',
    'apple': 'яблоко',
    'sandwich': 'сэндвич',
    'orange': 'апельсин',
    'broccoli': 'брокколи',
    'carrot': 'морковь',
    'hot dog': 'хот-дог',
    'pizza': 'пицца',
    'donut': 'пончик',
    'cake': 'торт',
    'chair': 'стул',
    'couch': 'диван',
    'potted plant': 'растение в горшке',
    'bed': 'кровать',
    'dining table': 'обеденный стол',
    'toilet': 'унитаз',
    'tv': 'телевизор',
    'laptop': 'ноутбук',
    'mouse': 'мышь',
    'remote': 'пульт',
    'keyboard': 'клавиатура',
    'cell phone': 'телефон',
    'microwave': 'микроволновка',
    'oven': 'духовка',
    'toaster': 'тостер',
    'sink': 'раковина',
    'refrigerator': 'холодильник',
    'book': 'книга',
    'clock': 'часы',
    'vase': 'ваза',
    'scissors': 'ножницы',
    'teddy bear': 'плюшевый мишка',
    'hair drier': 'фен',
    'toothbrush': 'зубная щетка'
}


class ObjectDetector:
    """Класс для распознавания объектов на изображениях"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Инициализация детектора объектов
        
        Параметры:
        - model_path: путь к модели YOLO (по умолчанию yolov8n.pt)
        """
        print("Загрузка модели YOLO...")
        self.model = YOLO(model_path)
        print("✓ Модель загружена успешно!")
    
    def load_image(self, image_path_or_url):
        """
        Загрузка изображения из файла или URL
        
        Параметры:
        - image_path_or_url: путь к файлу или URL изображения
        
        Возвращает:
        - PIL Image объект или None в случае ошибки
        """
        try:
            if image_path_or_url.startswith('http'):
                r = requests.get(image_path_or_url, timeout=10)
                img = Image.open(BytesIO(r.content)).convert('RGB')
            else:
                img = Image.open(image_path_or_url).convert('RGB')
            return img
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None
    
    def detect_objects(self, image_path, confidence=0.4):
        """
        Детекция объектов на изображении
        
        Параметры:
        - image_path: путь к изображению или URL
        - confidence: порог уверенности (0-1)
        
        Возвращает:
        - results: результаты детекции YOLO
        """
        results = self.model.predict(image_path, conf=confidence, verbose=False)
        return results
    
    def format_text_for_speech(self, object_names, language='ru'):
        """
        Формирование текста для озвучивания
        
        Параметры:
        - object_names: список названий объектов
        - language: язык ('ru' - русский, 'en' - английский)
        
        Возвращает:
        - text: сформированный текст
        """
        # Подсчитываем объекты
        object_counts = Counter(object_names)
        
        # Переводим на русский и формируем список
        objects_ru = []
        for obj, count in object_counts.items():
            if language == 'ru':
                obj_ru = TRANSLATION_DICT.get(obj, obj)
            else:
                obj_ru = obj
            
            if count > 1:
                objects_ru.append(f"{count} {obj_ru}")
            else:
                objects_ru.append(obj_ru)
        
        # Формируем финальный текст
        if len(objects_ru) == 0:
            text = "На изображении не обнаружено объектов"
        elif len(objects_ru) == 1:
            text = f"На этом изображении {objects_ru[0]}"
        elif len(objects_ru) == 2:
            text = f"На этом изображении {objects_ru[0]} и {objects_ru[1]}"
        else:
            text = f"На этом изображении " + ", ".join(objects_ru[:-1]) + f" и {objects_ru[-1]}"
        
        return text
    
    def generate_speech(self, text, language='ru', output_file='output_speech.mp3'):
        """
        Генерация аудио из текста
        
        Параметры:
        - text: текст для озвучивания
        - language: язык ('ru' - русский, 'en' - английский)
        - output_file: путь к выходному файлу
        
        Возвращает:
        - путь к файлу или None в случае ошибки
        """
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(output_file)
            return output_file
        except Exception as e:
            print(f"Ошибка синтеза речи: {e}")
            return None
    
    def process_image(self, image_path, confidence=0.4, language='ru', generate_audio=True):
        """
        Полная обработка изображения: детекция и озвучивание
        
        Параметры:
        - image_path: путь к изображению или URL
        - confidence: порог уверенности (0-1)
        - language: язык озвучивания ('ru' - русский, 'en' - английский)
        - generate_audio: генерировать ли аудио файл
        
        Возвращает:
        - dict с результатами обработки
        """
        # Загрузка изображения
        img = self.load_image(image_path)
        if img is None:
            return None
        
        # Детекция объектов
        results = self.detect_objects(image_path, confidence)
        detections = results[0]
        boxes = detections.boxes
        
        # Извлечение информации об объектах
        object_names = []
        detections_info = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = detections.names[cls_id]
            object_names.append(name)
            detections_info.append({
                'name': name,
                'name_ru': TRANSLATION_DICT.get(name, name),
                'confidence': conf,
                'bbox': box.xyxy[0].cpu().numpy().tolist()
            })
        
        # Формирование текста
        text = self.format_text_for_speech(object_names, language)
        
        # Генерация аудио
        audio_file = None
        if generate_audio and len(boxes) > 0:
            audio_file = self.generate_speech(text, language)
        
        # Получение аннотированного изображения
        annotated_img = detections.plot()
        
        return {
            'image': img,
            'annotated_image': annotated_img,
            'detections': detections_info,
            'text': text,
            'audio_file': audio_file,
            'total_objects': len(boxes)
        }
    
    def get_statistics(self, image_path, confidence=0.4):
        """
        Получение статистики по обнаруженным объектам без озвучивания
        
        Параметры:
        - image_path: путь к изображению или URL
        - confidence: порог уверенности (0-1)
        
        Возвращает:
        - dict со статистикой
        """
        results = self.detect_objects(image_path, confidence)
        boxes = results[0].boxes
        
        object_names = []
        confidences = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = results[0].names[cls_id]
            object_names.append(name)
            confidences.append(conf)
        
        object_counts = Counter(object_names)
        
        stats = {
            'total_objects': len(boxes),
            'unique_classes': len(object_counts),
            'object_counts': dict(object_counts),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0
        }
        
        return stats

