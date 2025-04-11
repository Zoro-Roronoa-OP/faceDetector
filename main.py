# Импорт нужных файлов/библиотек
import config # Файл с основными параметрами
import cv2 # библиотека для работы с видео (opencv)
from PIL import Image # Библиотека для работы с изображениями (Pillow)
from datetime import datetime # Библиотека для работы с датой и временем
import numpy as np # Библиотека для работы с массивами и математическими операциями
from deepface import DeepFace # Библиотека для анализа лиц
import telebot # Библиотека для работы с Telegram API
from sklearn.metrics.pairwise import cosine_similarity # Вычисление косинусного сходства


# Инициализация Telegram-бота с ключом API из файла config
bot = telebot.TeleBot(config.API_KEY)


# Функция для отправки изображения с надписью в Telegram
def send_image(img, current_time = None):

    if current_time is None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Форматируем время

    caption = f"Обнаружена неизвестная личность! Время {current_time}" # Текст сообщения
    # Конвертируем изображение из BGR (OpenCV) в RGB (Pillow)
    PIL_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

    bot.send_photo(config.CHAT_ID, PIL_image, caption=caption) # Отправляем фото в указанный чат


def get_embedding(img):

    """
    Генерирует эмбеддинг (векторное представление) лица
    
    Параметры:
        img: изображение с лицом
        
    Возвращает:
        numpy array с эмбеддингом или None, если лицо не найдено
    """

    try:
        result = DeepFace.represent(
            img_path = img,
            enforce_detection =False,  # Не выдавать ошибку, если лицо не найдено
            detector_backend = config.BACKEND,  # Детектор для поиска лица
            model_name = config.MODEL  # Модель для генерации эмбеддинга
        )
        return np.array(result[0]["embedding"]).reshape(1, -1) if result else None
    except Exception as e:
        return None


# Переменная для хранения последнего эмбеддинга
last_embedding = None

VIDEO_FILE = config.VIDEOFILE # Путь к видеофайлу (настраивается в файле config)

# Инициализация видеозахвата
video_capture = cv2.VideoCapture(VIDEO_FILE)
frame_count = 0 # Счётчик кадров

try:
    # Основной цикл для обработки видео
    while video_capture.isOpened():
        ret, frame = video_capture.read() # Читаем отдельный кадр

        frame_count += 1 # Увеличение счётчика кадров

        if frame_count % config.FRAMESKIP == 0: # Отбираем каждый n-ный кадр (n выставляется в файле config)
            try: 
                # Извлечение лиц из кадра с помощью библиотеки DeepFace
                face_objs = DeepFace.extract_faces( 
                    img_path = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    detector_backend = config.BACKEND,
                )


                # Получаем координаты лица (конкретный прямоугольник)
                x = face_objs[0]["facial_area"]["x"]
                y = face_objs[0]["facial_area"]["y"]
                x1 = face_objs[0]["facial_area"]["x"] + face_objs[0]["facial_area"]["w"]
                y1 = face_objs[0]["facial_area"]["y"] + face_objs[0]["facial_area"]["h"]
                width = face_objs[0]["facial_area"]["w"]
                height = face_objs[0]["facial_area"]["h"]
                

                # Проверяем, достаточного ли размера наше лицо
                if width >= config.MIN_FACE and height >= config.MIN_FACE:
                    
                    # Отрисовка красного прямоугольника вокург лица
                    frame = cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

                    # Обновляем текущий эмбеддинг
                    current_embedding = get_embedding(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if current_embedding is None:
                        continue # Пропуск кадра, если не удалось получить эмбеддинг

                    # Переменная для определения нового лица
                    is_new_face = True

                    if last_embedding is not None:
                        # Сравнение с последним известным лицом
                        sim = cosine_similarity(last_embedding, current_embedding)[0][0]
                        is_new_face = sim < config.SIMILARITY_THRESHOLD

                    if is_new_face: # Если лицо новое

                        # Поиск лица в базе данных (в белом списке)

                        dfs = DeepFace.find( 
                            img_path = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            model_name = config.MODEL,
                            detector_backend = config.BACKEND,
                            db_path = config.WHITELIST, # Белый список с фотографиями жильцов
                            threshold= config.THRESHOLD, # Порог схожести
                            silent = True, # Не выводить параметры поиска в консоль
                            distance_metric = config.METRIC # Метрика сравнения
                        )

                        # Если лицо не найдено в списке
                        is_blacklist = (dfs[0].identity.count() == 0)
                    
                        if is_blacklist: 
                            print("Обнаружено лицо! Не найдено в базе данных")
                            send_image(frame) # Отсылам фотографию в Telegram
                        else:
                            print("Обнаружено лицо! Найдено в базе данных")


                        # Обновляем последний эмбеддинг
                        last_embedding = current_embedding 

                
            except ValueError: # Исключение ошибки, когда на кадре нету лица
                pass
        
        # Добавление номера кадра на изображение (для наглядности и отладки)
        cv2.putText(frame, str(frame_count), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 

        # Отображение кадра в окне
        cv2.imshow('frame', frame)
            
        if cv2.waitKey(25) == 13:  # Нажатие Enter для выхода из цикла
            break
except cv2.error as e:
    print("Video end")
    pass
    
# Освобождаем ресурсы
video_capture.release()  # Освобождение видеозахвата
cv2.destroyAllWindows()  # Закрытие всех окон OpenCV