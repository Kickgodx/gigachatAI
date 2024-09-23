import pyaudio
import wave
import json
import threading
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from vosk import Model, KaldiRecognizer
from gigachat_bot import GigaChatBot
from gigachat_creds import auth_data, modelGigaChat

# Создание экземпляра GigaChatBot
bot = GigaChatBot(credentials=auth_data, model=modelGigaChat)

# Конфигурация записи аудио
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "./outputs/question.wav"
RESPONSE_OUTPUT_FILENAME = "./outputs/responses.txt"

# Загрузка модели Vosk
model = Model("./vosk-model-ru-0.42")  # Укажите путь к распакованной модели Vosk

print("Vosk model loaded successfully.")
print("GigaChat Audio Recorder is ready to use.")

# Блокировка для предотвращения одновременной записи
recording_lock = threading.Lock()

# Флаги для отслеживания состояния записи
is_recording_mic = False
is_recording_computer = False

# Переменные для хранения потоков записи
mic_stream = None
computer_stream = None
frames = []

def start_recording(audio, stream):
    global frames
    frames = []
    stream.start_stream()
    print("Recording started...")

def stop_recording(audio, stream):
    global frames
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Recording stopped.")

    # Сохранение записанного аудио в файл
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Нормализация уровня звука
    normalize_audio(WAVE_OUTPUT_FILENAME)

    # Транскрипция аудио и вывод текста
    transcribed_text = transcribe_audio(WAVE_OUTPUT_FILENAME)
    print(f"Transcribed text: {transcribed_text}")
    
    # Получение ответа от GigaChat
    response = bot.get_response(transcribed_text)
    print(f"AI response: {response}")

    # Запись ответа в файл
    with open(RESPONSE_OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
        f.write(f"User: {transcribed_text}\n")
        f.write(f"GigaChat: {response}\n\n")

    # Обновление текстового поля в интерфейсе
    update_text_widget(transcribed_text, response)

def normalize_audio(file_path):
    # Открытие аудиофайла
    wf = wave.open(file_path, 'rb')
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()

    # Чтение аудиоданных
    audio_data = wf.readframes(n_frames)
    wf.close()

    # Преобразование аудиоданных в массив numpy
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Нормализация аудиоданных
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = audio_array * (32767 / max_val)

    # Преобразование массива numpy обратно в байты
    normalized_audio_data = audio_array.astype(np.int16).tobytes()

    # Сохранение нормализованных аудиоданных в файл
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(n_channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(normalized_audio_data)
    wf.close()

def record_audio_from_mic():
    global is_recording_mic, mic_stream, frames
    with recording_lock:
        audio = pyaudio.PyAudio()
        if not is_recording_mic:
            # Настройка потока записи
            mic_stream = audio.open(format=FORMAT, channels=CHANNELS,
                                    rate=RATE, input=True,
                                    frames_per_buffer=CHUNK,
                                    stream_callback=callback)
            start_recording(audio, mic_stream)
            is_recording_mic = True
        else:
            stop_recording(audio, mic_stream)
            is_recording_mic = False

def record_audio_from_computer():
    global is_recording_computer, computer_stream, frames
    with recording_lock:
        audio = pyaudio.PyAudio()
        if not is_recording_computer:
            # Получение списка устройств ввода
            input_device_index = None
            for i in range(audio.get_device_count()):
                dev = audio.get_device_info_by_index(i)
                # Преобразование имени устройства из windows-1251 в utf-8
                try:
                    dev_name = dev['name'].encode('windows-1251').decode('utf-8')
                except UnicodeDecodeError:
                    dev_name = dev['name']
                if dev_name.lower().startswith("стерео") or dev_name.startswith("Стерео"):
                    input_device_index = i
                    break

            if input_device_index is None:
                print("Stereo Mix device not found.")
                return

            # Настройка потока записи
            computer_stream = audio.open(format=FORMAT, channels=CHANNELS,
                                         rate=RATE, input=True,
                                         input_device_index=input_device_index,
                                         frames_per_buffer=CHUNK,
                                         stream_callback=callback)
            start_recording(audio, computer_stream)
            is_recording_computer = True
        else:
            stop_recording(audio, computer_stream)
            is_recording_computer = False

def callback(in_data, frame_count, time_info, status):
    global frames
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)

def transcribe_audio(file_path):
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    print("Starting transcription...")
    result_text = ""

    while True:
        data = wf.readframes(CHUNK)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result = rec.FinalResult()
    result_dict = json.loads(result)
    result_text = result_dict.get('text', '')
    return result_text

def update_text_widget(transcribed_text, response):
    text_widget.config(state=tk.NORMAL)
    text_widget.insert(tk.END, f"User: {transcribed_text}\n")
    text_widget.insert(tk.END, f"GigaChat: {response}\n\n")
    text_widget.config(state=tk.DISABLED)

def start_mic_recording():
    threading.Thread(target=record_audio_from_mic).start()

def start_computer_recording():
    threading.Thread(target=record_audio_from_computer).start()

# Создание основного окна приложения
root = tk.Tk()
root.title("GigaChat Audio Recorder")

# Создание кнопок для управления записью
mic_button = tk.Button(root, text="Record from Microphone", command=start_mic_recording)
mic_button.pack(pady=10)

computer_button = tk.Button(root, text="Record from Computer", command=start_computer_recording)
computer_button.pack(pady=10)

# Создание текстового поля для отображения транскрибированного текста и ответа
text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=60, height=20)
text_widget.pack(pady=10)

# Запуск основного цикла приложения
root.mainloop()