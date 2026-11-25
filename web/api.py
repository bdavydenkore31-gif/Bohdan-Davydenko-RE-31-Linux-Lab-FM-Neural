"""Локальна веблабораторна розпізнавання FM серед інших сигналів."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile
import uuid
import wave
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from ai.inference import detect
from config import DATA_DIR, MODEL_DIR, STATIC_DIR
from drivers.iq_wav import IQFile

# Максимальний розмір завантаженого запису — 4 ГіБ.
MAX_BYTES=4*1024**3


# Фабрика застосунку FastAPI: усередині — маршрути й стан фонових аналізів.
def create_app():
    # tasks — активні фонові задачі. jobs — стан кожного аналізу за id.
    # gate — замок: одночасно рахується лише один аналіз (обчислення важкі).
    tasks=set()
    jobs={}
    gate=asyncio.Lock()

    # При вимкненні сервера дочікуємося завершення поточних аналізів.
    @asynccontextmanager
    async def lifespan(app):
        yield
        if tasks:
            await asyncio.gather(*tasks)

    app=FastAPI(title='Нейромережеве розпізнавання FM',lifespan=lifespan)

    # Статичні файли інтерфейсу. no-cache — щоб після змін браузер не брав стару версію.
    @app.get('/')
    async def index():
        return FileResponse(STATIC_DIR/'index.html',headers={'Cache-Control':'no-cache'})

    @app.get('/lab.js')
    async def script():
        return FileResponse(STATIC_DIR/'lab.js',media_type='text/javascript',headers={'Cache-Control':'no-cache'})

    @app.get('/style.css')
    async def stylesheet():
        return FileResponse(STATIC_DIR/'style.css',media_type='text/css',headers={'Cache-Control':'no-cache'})

    # Інформація про навчену модель (training.json) для інтерфейсу.
    @app.get('/api/model')
    async def metadata():
        path=MODEL_DIR/'training.json'
        if not path.is_file():
            raise HTTPException(404,'Модель ще не навчено. Запусти python -m ai.train.')
        return json.loads(path.read_text(encoding='utf-8'))

    # Фонове виконання аналізу: оновлює стан у jobs, зберігає результат у result.json,
    # а наприкінці видаляє завантажений файл.
    async def run(key,path,center,folder,raw_rate,raw_format):
        def progress(percent,text):
            jobs[key].update(progress=percent,message=text)
        try:
            # Черга: gate пускає по одному аналізу. Важка detect працює в окремому потоці, щоб не блокувати сервер.
            async with gate:
                jobs[key]['state']='running'
                result=await asyncio.to_thread(detect,path,center,MODEL_DIR,progress,raw_rate_hz=raw_rate,raw_format=raw_format)
                (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
                jobs[key].update(state='ready',progress=100,message='Аналіз готовий')
        # Помилки даних/файлу показуємо користувачеві.
        except (OSError,ValueError,RuntimeError) as exc:
            logging.exception('Аналіз %s',key)
            jobs[key].update(state='failed',message=str(exc))
        finally:
            path.unlink(missing_ok=True)

    # Приймає файл тілом запиту (потоково), перевіряє його й запускає аналіз у фоні. Повертає id задачі.
    @app.post('/api/analyze',status_code=202)
    async def analyze(request:Request,name:str,center_mhz:float|None=None,
                      raw_rate_hz:int|None=None,raw_format:Literal['u8','i16','f32']='u8'):
        # Перевірки вхідних даних: безпечне ім'я файлу (без шляхів), розширення, діапазони частот, розмір.
        if '/' in name or '\\' in name or not name.lower().endswith(('.wav','.iq','.raw','.bin')) or len(name)>220:
            raise HTTPException(422,'Вибери WAV або файл сирих I/Q із коректною назвою.')
        if raw_rate_hz is not None and not 320000<=raw_rate_hz<=5000000:
            raise HTTPException(422,'Частота відліків має бути від 0,32 до 5 МГц.')
        if center_mhz is not None and (not math.isfinite(center_mhz) or not 0<center_mhz<=100000):
            raise HTTPException(422,'Центральна частота має бути додатним числом у МГц.')
        try:
            size=int(request.headers.get('content-length',0))
        except ValueError as exc:
            raise HTTPException(400,'Некоректний розмір файлу.') from exc
        if size>MAX_BYTES:
            raise HTTPException(413,'Вибери файл до 4 ГіБ.')
        # Обмежуємо кількість одночасних задач.
        if len(tasks)>=3:
            raise HTTPException(409,'Дочекайся завершення поточного аналізу.')
        if not (MODEL_DIR/'classifier.npz').is_file():
            raise HTTPException(409,'Спочатку навчи модель командою python -m ai.train.')
        # Для кожного аналізу — окрема тимчасова папка.
        DATA_DIR.mkdir(parents=True,exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='analysis-',dir=DATA_DIR))
        path=folder/name
        try:
            # Читаємо тіло запиту блоками й пишемо на диск з перевіркою ліміту (весь файл у пам'ять не вантажимо).
            received=0
            with path.open('wb') as target:
                async for block in request.stream():
                    received+=len(block)
                    if received>MAX_BYTES:
                        raise HTTPException(413,'Вибери файл до 4 ГіБ.')
                    await asyncio.to_thread(target.write,block)
            if size and size!=received:
                raise ValueError('Передавання WAV перервано.')
            # Швидка перевірка, що файл справді є коректним I/Q і що відомий центр.
            iq=await asyncio.to_thread(IQFile,path,raw_rate_hz=raw_rate_hz,raw_format=raw_format)
            if center_mhz is None and iq.center_mhz is None:
                raise ValueError('Укажи центр I/Q у МГц або частоту в назві WAV.')
        except (ValueError,OSError,EOFError,wave.Error) as exc:
            shutil.rmtree(folder,ignore_errors=True)
            raise HTTPException(422,str(exc) or 'Файл не містить коректного I/Q WAV.') from exc
        # При будь-якій іншій помилці чистимо папку й передаємо помилку далі.
        except BaseException:
            shutil.rmtree(folder,ignore_errors=True)
            raise
        # Реєструємо задачу й запускаємо її у фоні.
        key=uuid.uuid4().hex
        jobs[key]={'state':'queued','progress':0,'message':'Чекаю на початок аналізу','folder':folder}
        task=asyncio.create_task(run(key,path,center_mhz,folder,raw_rate_hz,raw_format))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {'id':key}

    # Опитування стану аналізу (інтерфейс викликає періодично для індикатора прогресу).
    @app.get('/api/jobs/{key}')
    async def status(key:str):
        if key not in jobs:
            raise HTTPException(404,'Аналіз не знайдено.')
        return {k:v for k,v in jobs[key].items() if k!='folder'}

    # Готовий результат аналізу у форматі JSON.
    @app.get('/api/jobs/{key}/result')
    async def result(key:str):
        if key not in jobs or jobs[key]['state']!='ready':
            raise HTTPException(404,'Результат ще готується.')
        return FileResponse(jobs[key]['folder']/'result.json',media_type='application/json')

    return app


