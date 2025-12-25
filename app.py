"""
Streamlit приложение для распознавания объектов с озвучиванием
"""

import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
from object_detection import ObjectDetector, TRANSLATION_DICT

# Настройка страницы
st.set_page_config(
    page_title="Распознавание объектов с озвучиванием",
    page_icon="🔍",
    layout="wide"
)

# Инициализация детектора (кэширование)
@st.cache_resource
def load_detector(model_path='yolov8n.pt'):
    """Загрузка модели YOLO с кэшированием"""
    return ObjectDetector(model_path)

# Заголовок приложения
st.title("🔍 Распознавание объектов с озвучиванием")
st.markdown("---")
st.markdown("""
Это приложение использует YOLOv8 для распознавания объектов на изображениях 
и озвучивает результаты на русском языке с помощью gTTS.
""")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели
    model_option = st.selectbox(
        "Модель YOLO",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="n - nano (быстрая), s - small, m - medium, l - large, x - xlarge (точная)"
    )
    
    # Порог уверенности
    confidence = st.slider(
        "Порог уверенности",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Минимальная уверенность для детекции объекта"
    )
    
    # Язык озвучивания
    language = st.selectbox(
        "Язык озвучивания",
        ["ru", "en"],
        format_func=lambda x: "Русский" if x == "ru" else "English"
    )
    
    # Генерация аудио
    generate_audio = st.checkbox("Генерировать аудио", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Информация")
    st.info("""
    **Поддерживаемые форматы:**
    - JPG, PNG, JPEG
    - URL изображений
    
    **Возможности:**
    - Распознавание до 80 классов объектов
    - Озвучивание результатов
    - Визуализация с bounding boxes
    """)

# Загрузка детектора
try:
    detector = load_detector(model_option)
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# Основная область
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    # Способы загрузки
    upload_method = st.radio(
        "Способ загрузки",
        ["Загрузить файл", "Ввести URL"],
        horizontal=True
    )
    
    image_file = None
    image_url = None
    
    if upload_method == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'jpeg', 'png'],
            help="Загрузите изображение для распознавания"
        )
        if uploaded_file is not None:
            image_file = uploaded_file
    else:
        image_url = st.text_input(
            "URL изображения",
            placeholder="https://example.com/image.jpg",
            help="Введите URL изображения"
        )
        if image_url:
            image_file = image_url
    
    # Кнопка обработки
    process_button = st.button("🚀 Распознать объекты", type="primary", use_container_width=True)

with col2:
    st.header("📋 Результаты")
    
    if process_button and image_file:
        with st.spinner("Обработка изображения..."):
            try:
                # Сохранение загруженного файла во временный файл
                if upload_method == "Загрузить файл":
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image_path = temp_path
                else:
                    image_path = image_url
                
                # Обработка изображения
                result = detector.process_image(
                    image_path,
                    confidence=confidence,
                    language=language,
                    generate_audio=generate_audio
                )
                
                # Удаление временного файла
                if upload_method == "Загрузить файл" and os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if result is None:
                    st.error("Ошибка обработки изображения")
                else:
                    # Отображение результатов
                    if result['total_objects'] == 0:
                        st.warning("⚠️ Объекты не обнаружены на изображении")
                        st.image(result['image'], caption="Исходное изображение", use_container_width=True)
                    else:
                        # Статистика
                        st.success(f"✅ Найдено объектов: {result['total_objects']}")
                        
                        # Аннотированное изображение
                        annotated_rgb = cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="Распознанные объекты", use_container_width=True)
                        
                        # Таблица обнаруженных объектов
                        st.subheader("📊 Детали распознавания")
                        detections_data = []
                        for det in result['detections']:
                            detections_data.append({
                                'Объект (EN)': det['name'],
                                'Объект (RU)': det['name_ru'],
                                'Уверенность': f"{det['confidence']:.2%}"
                            })
                        
                        st.dataframe(detections_data, use_container_width=True, hide_index=True)
                        
                        # Текст для озвучивания
                        st.subheader("🔊 Текст озвучивания")
                        st.info(result['text'])
                        
                        # Аудио файл
                        if result['audio_file'] and os.path.exists(result['audio_file']):
                            st.subheader("🎵 Аудио")
                            with open(result['audio_file'], "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format='audio/mp3')
                            
                            # Кнопка скачивания аудио
                            st.download_button(
                                label="📥 Скачать аудио",
                                data=audio_bytes,
                                file_name="output_speech.mp3",
                                mime="audio/mp3"
                            )
                        
            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.exception(e)
    
    elif process_button:
        st.warning("⚠️ Пожалуйста, загрузите изображение или введите URL")

# Дополнительная информация
st.markdown("---")
with st.expander("ℹ️ О проекте"):
    st.markdown("""
    ### Описание
    Это приложение использует современные технологии машинного обучения для распознавания объектов:
    
    - **YOLOv8**: Модель детекции объектов в реальном времени
    - **gTTS**: Google Text-to-Speech для синтеза речи
    - **Streamlit**: Веб-интерфейс для удобного использования
    
    ### Поддерживаемые классы объектов
    Приложение может распознавать 80 различных классов объектов, включая:
    - Люди, животные
    - Транспорт (автомобили, велосипеды, самолеты и т.д.)
    - Мебель и бытовая техника
    - Еда и напитки
    - Спортивный инвентарь
    - И многое другое
    
    ### Технологии
    - Python 3.8+
    - PyTorch
    - Ultralytics YOLO
    - Streamlit
    - OpenCV
    """)

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Распознавание объектов с озвучиванием | Powered by YOLOv8 & Streamlit"
    "</div>",
    unsafe_allow_html=True
)

