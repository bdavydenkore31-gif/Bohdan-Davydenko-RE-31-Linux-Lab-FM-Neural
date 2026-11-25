"""Комплексний I/Q у PCM WAV або сирих парах u8, i16 чи float32."""
from pathlib import Path
import re
import struct
import wave

import numpy as np


# Читач комплексного I/Q-запису: WAV (2 канали = I і Q) або «сирий» файл без заголовка.
# Читає шматками за потреби й не вантажить файл у пам'ять повністю (файли можуть бути до 4 ГіБ).
class IQFile:
    def __init__(self, path, *, raw_rate_hz=None, raw_format='u8'):
        self.path = Path(path)
        # Сирий режим: заголовка немає, тому частоту й формат відліків задає користувач.
        self.raw=raw_rate_hz is not None
        self.sample_format=raw_format
        if self.raw:
            if not isinstance(raw_rate_hz,(int,np.integer)):
                raise ValueError('Частота відліків має бути цілим числом у Гц.')
            if raw_format not in ('u8','i16','f32'):
                raise ValueError('Вибери формат відліків I/Q.')
            self.channels=2
            self.width={'u8':1,'i16':2,'f32':4}[raw_format]
            self.rate=raw_rate_hz
            size=self.path.stat().st_size
            # Розмір файлу має ділитися на розмір пари I/Q.
            if size%(2*self.width):
                raise ValueError('Розмір файлу не відповідає вибраному формату пар I/Q.')
            self.frames=size//(2*self.width)
        else:
            # WAV: перевіряємо сигнатуру RIFF/WAVE, потім читаємо параметри стандартним модулем wave.
            try:
                with self.path.open('rb') as source:
                    header=source.read(12)
                if len(header)<12:
                    raise ValueError('Файл надто короткий для WAV. Якщо це сирий I/Q, укажи його параметри.')
                if header[:4]!=b'RIFF' or header[8:12]!=b'WAVE':
                    raise ValueError('Заголовок WAV відсутній. Для сирого I/Q вкажи частоту відліків і формат.')
                with wave.open(str(path),'rb') as wav:
                    self.channels,self.width,self.rate,self.frames,*_=wav.getparams()
                    # Перевірка, що файл не обірваний: читаємо останній кадр.
                    if self.frames:
                        wav.setpos(self.frames-1)
                        if len(wav.readframes(1))!=self.channels*self.width:
                            raise ValueError('Файл WAV обірвано.')
            except (wave.Error,EOFError) as exc:
                raise ValueError('Некоректний WAV. Потрібен двоканальний PCM I/Q, 8/16/32 біти.') from exc
        # Загальна перевірка: два канали, 8/16/32 біти, частота 0,32–5 МГц.
        if self.channels!=2 or self.width not in (1,2,4) or not 320000<=self.rate<=5000000:
            raise ValueError('Потрібні два канали I/Q, 8/16/32 біти та частота відліків від 0,32 до 5 МГц.')
        # Тривалість = кількість кадрів / частота відліків.
        self.duration=self.frames/self.rate
        if not .05<=self.duration<=600:
            raise ValueError('Підтримується I/Q-запис тривалістю від 0,05 до 600 с. Перевір частоту відліків.')
        # Центральну частоту беремо з назви файлу, напр. baseband_96400000Hz_….wav → 96,4 МГц.
        match = re.search(r'(?:^|[_-])(\d{7,10})\s*Hz(?:[_\-.]|$)',self.path.name,re.I)
        self.center_mhz = int(match[1])/1e6 if match else None
        if self.center_mhz is None and not self.raw:
            # SDR# зберігає центральну частоту в додатковому WAV-блоці auxi.
            with self.path.open('rb') as stream:
                stream.seek(12)
                while stream.tell()+8<=self.path.stat().st_size:
                    tag,size=struct.unpack('<4sI',stream.read(8))
                    if tag==b'data':
                        break
                    if tag==b'auxi' and size>=36:
                        payload=stream.read(36)
                        if len(payload)==36:
                            center=struct.unpack_from('<I',payload,32)[0]
                            if center>0:
                                self.center_mhz=center/1e6
                        break
                    stream.seek(size+(size%2),1)

    # Читає count комплексних відліків від позиції start (у кадрах). Повертає масив complex: I + j·Q.
    def read(self, start, count):
        # Затискаємо межі, щоб не вийти за файл.
        start=max(0,min(self.frames,start))
        count=max(0,min(count,self.frames-start))
        if self.raw:
            with self.path.open('rb') as source:
                source.seek(start*2*self.width)
                raw=source.read(count*2*self.width)
        else:
            with wave.open(str(self.path),'rb') as wav:
                wav.setpos(start)
                raw=wav.readframes(count)
        if len(raw)!=count*2*self.width:
            raise ValueError('Файл I/Q обірвано під час читання.')
        # Тип даних: float32 для 'f32', інакше ціле.
        dtype='<f4' if self.raw and self.sample_format=='f32' else {1:'u1',2:'<i2',4:'<i4'}[self.width]
        values=np.frombuffer(raw,dtype).astype(np.float32).reshape(-1,2)
        # Приводимо до діапазону [-1, 1): u8 — віднімаємо 128 (беззнакові), цілі зі знаком — ділимо на 2^(біти−1).
        if self.width==1:
            values=(values-128)/128
        elif dtype!='<f4':
            values/=2**(8*self.width-1)
        if not np.isfinite(values).all():
            raise ValueError('Відліки містять некоректні числа. Перевір формат I/Q.')
        return values[:,0]+1j*values[:,1]


# Записує комплексний сигнал у 16-бітний стерео WAV (I — лівий канал, Q — правий).
# Масштабує за піком до 0,9.
def write_iq(path, samples, rate):
    peak=max(float(np.max(np.abs(samples.real))),float(np.max(np.abs(samples.imag))),1e-8)
    scale=.9/peak
    stereo=np.column_stack((samples.real,samples.imag))
    with wave.open(str(path),'wb') as wav:
        wav.setparams((2,2,rate,0,'NONE','not compressed'))
        wav.writeframes(np.rint(stereo*scale*32767).astype('<i2').tobytes())
