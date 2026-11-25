"""Пошук модуляцій у часових і частотних вікнах I/Q WAV."""
import math

import numpy as np
from scipy import fft, ndimage

from ai.model import Classifier
from ai.features import RATE, SAMPLES
from drivers.iq_wav import IQFile
from processing.spectrum import channel, waterfall
from processing.report import summarize


# Шукає на широкосмуговому записі частотні зсуви, де є сигнал (кандидати в канали).
# Повертає список зсувів у Гц відносно центру запису, кратних 2,5 кГц.
def candidates(raw, rate):
    # Вузький запис уже є одним каналом, шукати нічого.
    if rate<=360000:
        return [0]
    # Усереднений спектр потужності: ріжемо на блоки по 512 відліків, FFT з вікном Ханна, усереднюємо.
    bins=512
    blocks=raw[:len(raw)//bins*bins].reshape(-1,bins)
    power=np.mean(np.abs(fft.fftshift(fft.fft(blocks*np.hanning(bins),axis=1),axes=1))**2,axis=0)
    # Прибираємо DC-пік посередині.
    power[bins//2]=(power[bins//2-1]+power[bins//2+1])/2
    # Згладжуємо спектр, щоб дрібні викиди не давали хибних кандидатів.
    smoothed=ndimage.gaussian_filter1d(power,1.5)
    # Рівень шуму оцінюємо як 20-й перцентиль спектра.
    floor=max(float(np.quantile(smoothed,.2)),1e-12)
    # «Активні» частоти — вищі за шум більш ніж у 4 рази. Закриваємо дрібні розриви всередині одного сигналу.
    active=smoothed>floor*4
    active=ndimage.binary_closing(active,structure=np.ones(5))
    # Нумеруємо суцільні смуги активних частот: кожна смуга — кандидат у канал.
    labels,total=ndimage.label(active)
    axis=fft.fftshift(fft.fftfreq(bins,1/rate))
    offsets=[]
    for index in range(1,total+1):
        inside=labels==index
        # Центр смуги — середня частота, зважена надлишком потужності над шумом.
        weights=np.maximum(power[inside]-floor,0)
        if not weights.sum():
            continue
        offset=float(np.sum(axis[inside]*weights)/weights.sum())
        # Беремо лише канали, що повністю вміщаються в запис. Округлюємо до 2,5 кГц.
        if abs(offset)<=rate/2-RATE/2:
            offsets.append(round(offset/2500)*2500)
    return offsets


# Головна функція аналізу: читає I/Q-файл, будує водоспад і для кожного півсекундного відрізка
# розпізнає FM / не FM у знайдених каналах.
# progress(відсоток, текст) викликається для індикатора прогресу у вебі.
def detect(path, center_mhz, model_dir, progress=lambda value,text:None, *, raw_rate_hz=None, raw_format='u8'):
    # Відкриваємо файл і завантажуємо модель.
    iq=IQFile(path,raw_rate_hz=raw_rate_hz,raw_format=raw_format)
    net=Classifier(model_dir)
    # Центральна частота: або задана вручну, або з назви/заголовка файлу.
    center=center_mhz if center_mhz is not None else iq.center_mhz
    if center is None or not math.isfinite(center) or center<=0:
        raise ValueError('Укажи центральну частоту I/Q-запису в МГц.')
    # Водоспад (спектр у часі) потрібен інтерфейсу для малювання.
    progress(5,'Будую водоспад…')
    rows=waterfall(iq)
    # Аналізуємо відрізками по 0,5 с. Для кожного беремо 0,2 с сирих відліків із середини.
    count=max(1,math.ceil(iq.duration/.5))
    raw_count=math.ceil(.20*iq.rate)
    found=[]
    for k in range(count):
        start=k*.5
        stop=min(iq.duration,start+.5)
        # Позиція 0,2-секундного шматка по центру відрізка (в межах файлу).
        position=max(0,min(iq.frames-raw_count,round((start+stop)/2*iq.rate)-raw_count//2))
        raw=iq.read(position,raw_count)
        # Знаходимо канали із сигналом у цьому шматку.
        offsets=candidates(raw,iq.rate)
        # Сигналів немає: оновлюємо прогрес і йдемо далі.
        if not offsets:
            progress(10+round(85*(k+1)/count),f'Розпізнаю сигнал · {k+1} / {count} часових вікон')
            continue
        # Для кожного каналу вирізаємо кілька (votes) вікон по 8192 відліки, щоб вирішувати голосуванням.
        windows=[]
        votes=net.training.get('inference_windows',1)
        for offset in offsets:
            # Переносимо канал у нуль і ресемплимо до 320 кГц.
            z=channel(raw,iq.rate,offset)
            for fraction in np.linspace(.2,.8,votes) if votes>1 else [.5]:
                skip=max(0,round((len(z)-SAMPLES)*fraction))
                windows.append(z[skip:skip+SAMPLES])
        # Прогноз для всіх вікон, форма [канал, вікно, клас].
        predictions=net.predict(windows).reshape(len(offsets),votes,-1)
        # Медіана трьох неперекривних вікон потребує двох узгоджених рішень.
        probabilities=np.median(predictions,axis=1)
        # Спочатку обробляємо найупевненіші канали.
        ordered=np.argsort(probabilities.max(axis=1))[::-1]
        centers=[]
        for index in ordered:
            # Клас з найвищою ймовірністю та його упевненість.
            label=int(np.argmax(probabilities[index]))
            kind=net.classes[label]
            score=float(probabilities[index,label])
            offset=float(offsets[index])
            # Поріг залежить від класу (для FM суворіший). Нижче порога рішення не приймаємо.
            threshold=net.training['threshold'] if kind=='fm' else net.training.get('non_fm_threshold',net.training['threshold'])
            # Відкидаємо невпевнені рішення й дублікати: канали ближче ніж 125 кГц до вже прийнятого вважаємо тим самим сигналом.
            if score<threshold or any(abs(offset-other)<125000 for other in centers):
                continue
            centers.append(offset)
            found.append({'class':kind,'frequency_mhz':round(center+offset/1e6,6),
                          'start_s':round(start,3),'end_s':round(stop,3),'score':round(score,4)})
        progress(10+round(85*(k+1)/count),f'Розпізнаю сигнал · {k+1} / {count} часових вікон')
    # Зливаємо сусідні за часом виявлення одного класу й тієї ж частоти в один проміжок.
    merged=[]
    for item in sorted(found,key=lambda x:x['start_s']):
        previous=next((x for x in reversed(merged) if x['class']==item['class']
                      and abs(x['end_s']-item['start_s'])<.002
                      and abs(x['frequency_mhz']-item['frequency_mhz'])<.05),None)
        if previous is None:
            merged.append(dict(item))
        else:
            previous['end_s']=item['end_s']
            # Для злитого проміжку беремо найнижчу упевненість (консервативно).
            previous['score']=min(previous['score'],item['score'])
    progress(100,'Готово')
    # Результат для веб-інтерфейсу: водоспад, виявлення, підсумок і метадані моделі.
    return {'filename':iq.path.name,'center_mhz':center,'sample_rate_hz':iq.rate,'duration_s':iq.duration,
            'rows':rows,'detections':merged,'summary':summarize(merged,iq.duration),
            'time_step_s':.5,'frequency_step_khz':2.5,
            'model_threshold':net.training['threshold'],'training':net.training}
