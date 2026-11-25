"""Вибір сирих FM I/Q-фрагментів із попереднього SDR-запису."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ai.features import RATE, SAMPLES
from config import DATASET_PATH
from drivers.iq_wav import IQFile
from processing.spectrum import channel


# SHA-256 файлу (читаємо блоками по 1 МБ). У метаданих видно, з якого саме запису зроблено набір.
def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            sha.update(block)
    return sha.hexdigest()


# Вирізає count рівномірно розподілених у часі вікон каналу станції з широкосмугового запису й зберігає в .npz.
def prepare(source, station, output, count=960):
    iq=IQFile(source)
    # Без центральної частоти не можна знайти потрібну станцію в спектрі.
    if iq.center_mhz is None:
        raise ValueError('Назва вихідного WAV має містити центр, наприклад baseband_96000000Hz_capture.wav.')
    # Зсув станції відносно центру запису (Гц). Перевіряємо, що канал не виходить за межі запису.
    offset=(station-iq.center_mhz)*1e6
    if abs(offset)+RATE/2>iq.rate/2:
        raise ValueError('Канал виходить за межі вихідного запису.')
    if iq.duration/count<.05:
        raise ValueError('Забагато фрагментів для цього запису. Зменш --windows.')
    values=[];starts=[];ends=[]
    # Скільки сирих відліків потрібно, щоб після ресемплінгу вийшло 8192 (з запасом на краї фільтра).
    raw_count=math.ceil((SAMPLES/RATE+.014)*iq.rate)
    # Беремо вікна рівномірно по всьому запису.
    for k in range(count):
        position=round((k+.5)*iq.frames/count)-raw_count//2
        # Переносимо станцію в нуль і зводимо до 320 кГц.
        z=channel(iq.read(position,raw_count),iq.rate,offset)
        # Відрізаємо краї (перехідні процеси фільтра), лишаємо центральні SAMPLES відліків.
        skip=(len(z)-SAMPLES)//2
        z=z[skip:skip+SAMPLES]
        if len(z)!=SAMPLES:
            raise ValueError('Не вдалося отримати повне I/Q-вікно.')
        # Однакова шкала для всіх вікон. Квантування зберігає I і Q.
        values.append(np.rint(np.clip(np.column_stack((z.real,z.imag)),-.999,.999)*32767).astype('<i2'))
        starts.append(position/iq.rate)
        ends.append((position+raw_count)/iq.rate)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Зберігаємо стиснутий .npz: I/Q, часові межі кожного вікна та частоту.
    np.savez_compressed(output,iq=np.asarray(values),start_s=np.asarray(starts),end_s=np.asarray(ends),rate=RATE)
    # Метадані (JSON поруч): звідки дані, як розмічено, як ділиться на train/validation/test.
    metadata={'source_filename':source.name,'source_sha256':digest(source),'source_sample_rate_hz':iq.rate,
              'source_center_mhz':iq.center_mhz,'station_mhz':station,'source_duration_s':iq.duration,
              'windows':count,'samples_per_window':SAMPLES,'window_rate_hz':RATE,
              'label_basis':f'Вибраний для навчання канал {station:g} МГц має ручну мітку FM. Початковий набір підготовлено з перевіреного каналу попередньої лабораторної.',
              'processing':'Зсув частоти, фільтрація й ресемплінг комплексних I/Q. Аудіо не декодується.',
              'split':'Різні вихідні часові вікна для train 70%, validation 15%, test 15%; їхні перетворення зберігають той самий split.'}
    output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output,output.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    # input — вихідний широкосмуговий WAV. --station-mhz — частота FM-станції, яку вирізаємо.
    parser.add_argument('input',type=Path)
    parser.add_argument('--station-mhz',type=float,default=96)
    parser.add_argument('--windows',type=int,default=960)
    parser.add_argument('--output',type=Path,default=DATASET_PATH)
    args=parser.parse_args()
    if args.windows<100:
        parser.error('Потрібні щонайменше 100 вихідних вікон.')
    prepare(args.input,args.station_mhz,args.output,args.windows)
