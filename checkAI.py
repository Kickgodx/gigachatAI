import pyaudio
import wave
import json
import keyboard
from vosk import Model, KaldiRecognizer
import threading
import time
import numpy as np
from gigachat_bot import GigaChatBot
from gigachat_creds import auth_data, modelGigaChat, modelGigaChat_Plus, modelGigaChat_Pro

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
    # print(f"Transcribed text: {transcribed_text}")
    
    # Получение ответа от GigaChat
    response = bot.get_response(transcribed_text)
    print(f"AI response: {response}")

    # Запись ответа в файл
    with open(RESPONSE_OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
        f.write(f"User: {transcribed_text}\n")
        f.write(f"GigaChat: {response}\n\n")

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
                if 'cтерео микшер' in dev['name'].lower() or 'wave out' in dev['name'].lower() or 'Стерео микшер' in dev['name'].lower():
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
    # print(f"Final result: {result}")
    result_dict = json.loads(result)
    result_text = result_dict.get('text', '')
    print(f"Transcribed text: {result_text}")
    return result_text

def handle_key_presses():
    while True:
        if keyboard.is_pressed('left'):
            record_audio_from_mic()
            time.sleep(0.5)  # Задержка для предотвращения многократного срабатывания
        elif keyboard.is_pressed('right'):
            record_audio_from_computer()
            time.sleep(0.5)  # Задержка для предотвращения многократного срабатывания
        elif keyboard.is_pressed('esc'):
            print("Exiting...")
            break

def main():
    print("Press the left arrow key to start/stop recording from the microphone.")
    print("Press the right arrow key to start/stop recording from the computer.")
    print("Press the Esc key to exit.")
    
    key_thread = threading.Thread(target=handle_key_presses)
    key_thread.start()
    key_thread.join()

if __name__ == "__main__":
    main()