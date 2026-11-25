"""Розмічені I/Q-вікна з FM, ADS-B, airband і міжканальних ділянок."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ai.features import RATE,SAMPLES
from config import DATASET_PATH
from drivers.iq_wav import IQFile
from processing.spectrum import channel


# Збирає розмічений набір із трьох записів (ADS-B, airband, broadcast) і старого FM-набору.
# Мітки: 'fm' — FM-станція, 'non_fm' — все інше.
# Кожне вікно має ще й split: 0 = train, 1 = validation, 2 = test.
def prepare(folder,output,windows=240):
    # Паралельні списки: i-й елемент кожного списку описує i-те вікно.
    values=[];labels=[];kinds=[];sources=[];splits=[];starts=[];ends=[];offsets=[]
    metadata=[]

    # Перевіряє вікно й додає його до списків.
    def append(z,label,kind,source,split,start,end,offset):
        if len(z)!=SAMPLES or not np.isfinite(z).all():
            raise ValueError('Неповне навчальне I/Q-вікно.')
        # Нормуємо за піком до 0,95 повного діапазону й зберігаємо як int16: файл виходить компактним.
        peak=max(float(np.max(np.abs(z.real))),float(np.max(np.abs(z.imag))),1e-9)
        samples=np.column_stack((z.real,z.imag))*(.95/peak)
        values.append(np.rint(samples*32767).astype('<i2'))
        labels.append(label);kinds.append(kind);sources.append(source);splits.append(split)
        starts.append(start);ends.append(end);offsets.append(offset)

    # Позитивні вікна попередньої лабораторної залишають початковий часовий split.
    with np.load(DATASET_PATH,allow_pickle=False) as old:
        n=len(old['iq'])
        # Беремо кожне четверте вікно (сусідні майже однакові). Split за часом: перші 70% — train, наступні 15% — validation, решта — test.
        for i in range(0,n,4):
            pair=old['iq'][i].astype(np.float32)/32768
            split=0 if i<round(.7*n) else 1 if i<round(.85*n) else 2
            append(pair[:,0]+1j*pair[:,1],'fm','real_fm','previous_fm',split,
                   float(old['start_s'][i]),float(old['end_s'][i]),0)
    metadata.append({'id':'previous_fm',**json.loads(DATASET_PATH.with_suffix('.json').read_text(encoding='utf-8'))})
    # Опис записів: (ідентифікатор, файл, частота відліків для сирого I/Q або None для WAV із заголовком, список каналів).
    # Канал = (зсув від центру у Гц, мітка, тип джерела).
    specs=[
        ('adsb','adsb.2021-11-26T15_03_30_573.wav',None,
         [(0,'non_fm','real_adsb'),(650000,'non_fm','real_adsb_background')]),
        ('airband','airband.2021-12-16T14_38_30_460.wav',None,
         [(-675000,'non_fm','real_airband'),(-545000,'non_fm','real_airband'),
          (1125000,'non_fm','real_airband')]),
        ('broadcast','baseband_96400000Hz_09-17-43_23-11-2026.wav',2048000,
         [(-400000,'fm','real_fm'),(0,'fm','real_fm'),
          (-150000,'non_fm','real_background'),(230000,'non_fm','real_background'),
          (580000,'non_fm','real_background')]),
    ]
    for ident,name,raw_rate,channels in specs:
        path=folder/name
        iq=IQFile(path,raw_rate_hz=raw_rate)
        print(f'{ident}: {iq.rate} Гц, {iq.duration:.2f} с',flush=True)
        # Часові частини розділяє захисна пауза 1 с. Сусідні канали мають той самий split.
        raw_count=round(.06*iq.rate)
        for split,(begin,end) in enumerate(((0,.7),(.7,.85),(.85,1))):
            # Межі частини в секундах. Відступ 0,5 с з обох боків — захисна пауза, щоб вибірки не перетиналися.
            begin=begin*iq.duration+.5;end=end*iq.duration-.5
            # Кількість точок пропорційна довжині частини.
            number=max(1,round(windows*(end-begin)/iq.duration))
            # У кожній точці часу беремо короткий шматок і вирізаємо з нього всі описані канали.
            for t in np.linspace(begin,end,number):
                position=round(t*iq.rate)-raw_count//2
                raw=iq.read(position,raw_count)
                for offset,label,kind in channels:
                    z=channel(raw,iq.rate,offset)
                    skip=(len(z)-SAMPLES)//2
                    append(z[skip:skip+SAMPLES],label,kind,ident,split,
                           position/iq.rate,(position+raw_count)/iq.rate,offset)
        # Контрольна сума файлу для відтворюваності.
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):
                sha.update(block)
        metadata.append({'id':ident,'filename':name,'sha256':sha.hexdigest(),
                         'sample_rate_hz':iq.rate,'raw_format':'u8' if raw_rate else 'PCM16 WAV',
                         'rate_basis':'Параметри попереднього записування; у сирому файлі заголовка немає.' if raw_rate else 'Заголовок WAV',
                         'duration_s':iq.duration,
                         'channels':[{'offset_hz':o,'class':l,'kind':k} for o,l,k in channels]})
    # Одне джерело не може передати той самий час до різних вибірок.
    for source in set(sources):
        for split in (0,1):
            left=[e for s,e,g in zip(sources,ends,splits) if s==source and g==split]
            right=[b for s,b,g in zip(sources,starts,splits) if s==source and g==split+1]
            if left and right and max(left)>=min(right):
                raise ValueError('Перетин часових частин даних.')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Зберігаємо набір: I/Q, мітки, типи, джерела, split, час і зсув кожного вікна.
    np.savez_compressed(output,iq=values,label=labels,kind=kinds,source=sources,split=splits,
                        start_s=starts,end_s=ends,offset_hz=offsets,rate=RATE)
    # Метадані набору: як розмічено, як розділено і який канал свідомо відкладено для чесної перевірки.
    result={'schema':3,'sources':metadata,'windows':len(values),'samples_per_window':SAMPLES,
            'sample_rate_hz':RATE,
            'label_basis':'FM попередньої лабораторної; у новому записі канали 96,0 і 96,4 МГц перевірено за фазовою структурою та пілотом 19 кГц. ADS-B і airband — негативи за типом наданих записів. Міжканальні ділянки мають мітку не FM у центрі аналізованого каналу.',
            'split':'70/15/15 за часом кожного запису, із захисною паузою; усі канали одного часу належать одній вибірці.',
            'held_out_channel':'Канал близько 97,2 МГц нового запису не використано для навчання або вибору моделі.'}
    output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{output}: {len(values)} вікон',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir',type=Path,required=True)
    parser.add_argument('--windows',type=int,default=240)
    parser.add_argument('--output',type=Path,default=DATASET_PATH.with_name('labeled_iq.npz'))
    args=parser.parse_args()
    if args.windows<100:
        parser.error('Потрібно щонайменше 100 часових точок на запис.')
    prepare(args.samples_dir,args.output,args.windows)
