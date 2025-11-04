import os
import time
import datetime
import threading
import requests
import schedule
import subprocess
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# Настройки
# ----------------------------
OUTPUT_DIR = "recordings"
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg.exe"

# ----------------------------
# Потоки: (URL, длительность в секундах, is_hls)
# is_hls = True → поток требует FFmpeg (например, m3u8)
# is_hls = False → прямой аудиопоток (HTTP/HTTPS), запись через requests
# ----------------------------
STREAMS = {
    "Qazaq_Radiosy": ("https://radio-streams.kaztrk.kz/qazradio/qazradio/icecast.audio", 360, False),
    "Shalqar_Radiosy": ("https://radio-streams.kaztrk.kz/shalqar/shalqar/icecast.audio", 360, False),
    "Classic_Radiosy": ("https://radio-streams.kaztrk.kz/classic/classic/icecast.audio", 300, False),
    "Energy_FM": ("http://89.219.34.117:8008/NEWENERGY", 600, False),
    "Zhuldyz_FM": ("http://91.201.214.229:8000/zhulduz", 300, False),
    "Toiduman_Radio": ("https://stream.gakku.kz:8443/live.mp3", 300, False),
    "Qazaqstan": (
        "https://qazaqstantv-stream.qazcdn.com/international/international/tracks-v1a1/mono.ts.m3u8", 300, True),
}

# ----------------------------
# Расписание
# ----------------------------
SCHEDULES = {
    "Qazaq_Radiosy": [8, 9, 11, 13, 15, 17],
    "Shalqar_Radiosy": [10, 12, 14, 16, 18],
    "Classic_Radiosy": [8, 12, 14, 18, 20],
    "Energy_FM": [11, 18],
    "Zhuldyz_FM": "hourly_from_0700_to_2400",
    "Toiduman_Radio": [9, 13, 18],
    "Qazaqstan": [(10, 780), (13, 780), (17, 1320), (20, 1920)],
}


# ----------------------------
# Функции записи
# ----------------------------
def record_http_stream(station_name: str, stream_url: str, duration_sec: int):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    station_dir = os.path.join(OUTPUT_DIR, date_str, station_name)
    os.makedirs(station_dir, exist_ok=True)
    filename = os.path.join(station_dir, f"{station_name}_{now.strftime('%Y%m%d_%H%M')}.mp3")

    print(f"[{now}] 📻 Начинаю запись: {station_name}, длительность: {duration_sec} с")

    try:
        with requests.get(stream_url, stream=True, timeout=duration_sec + 60, verify=False) as resp:
            resp.raise_for_status()
            start_time = time.time()
            with open(filename, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if time.time() - start_time > duration_sec:
                        break
                    if chunk:
                        f.write(chunk)
        print(f"[{datetime.datetime.now()}] Запись завершена: {filename}")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Ошибка записи {station_name}: {e}")


def record_hls_stream(station_name: str, stream_url: str, duration_sec: int):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    station_dir = os.path.join(OUTPUT_DIR, date_str, station_name)
    os.makedirs(station_dir, exist_ok=True)
    filename = os.path.join(station_dir, f"{station_name}_{now.strftime('%Y%m%d_%H%M')}.mp3")

    print(f"[{now}] 📻 Начинаю запись: {station_name}, длительность: {duration_sec} с")

    try:
        cmd = [
            FFMPEG_PATH,
            "-i", stream_url,
            "-t", str(duration_sec),
            "-vn",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-y",
            filename
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"[{datetime.datetime.now()}] Запись завершена: {filename}")
        else:
            print(f"[{datetime.datetime.now()}] Ошибка записи {station_name}: ffmpeg failed")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] Ошибка записи {station_name}: {e}")


# ----------------------------
# Планировщик
# ----------------------------
def schedule_station(station_name, rule):
    stream_data = STREAMS.get(station_name)
    if not stream_data:
        print(f"[WARN] Нет данных для станции: {station_name}")
        return

    url, base_duration, is_hls = stream_data
    record_func = record_hls_stream if is_hls else record_http_stream

    def job(duration_override=None):
        period = duration_override if duration_override is not None else base_duration
        threading.Thread(target=record_func, args=(station_name, url, period)).start()

    if isinstance(rule, list):
        if not rule:
            return
        if isinstance(rule[0], tuple):
            for hour, duration in rule:
                schedule.every().day.at(f"{int(hour):02d}:00").do(job, duration_override=duration)
        else:
            for hour in rule:
                schedule.every().day.at(f"{int(hour):02d}:00").do(job)
    elif rule == "hourly_from_0700_to_2400":
        for hour in range(7, 24):
            schedule.every().day.at(f"{hour:02d}:00").do(job)
    else:
        print(f"[WARN] Неизвестное правило расписания: {rule}")


# ----------------------------
# Инициализация
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for station, rule in SCHEDULES.items():
    schedule_station(station, rule)
    print(f"📅 Расписание добавлено для: {station}")

print("\n🚀 Планировщик запущен. Ожидаю времени записи...\n")

while True:
    schedule.run_pending()
    time.sleep(1)
