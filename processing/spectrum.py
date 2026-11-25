"""Частотний перенос, ресемплінг та спектрограма I/Q."""
import math

import numpy as np
from scipy import fft, signal


# Виділяє один канал із широкосмугового запису: зсуває частоту offset_hz у нуль і ресемплить до target_rate.
# Саме так із запису на 2 МГц виходить 320-кГц вікно однієї станції.
def channel(z, source_rate, offset_hz, target_rate=320000):
    n=np.arange(len(z),dtype=np.float64)
    # Множення на комплексну експоненту зсуває спектр: потрібна станція опиняється на 0 Гц.
    moved=z*np.exp(-2j*np.pi*offset_hz*n/source_rate).astype(np.complex64)
    # Ресемплінг у раціональне число разів. Поліфазний фільтр сам відсікає зайве поза каналом.
    divisor=math.gcd(source_rate,target_rate)
    return signal.resample_poly(moved,target_rate//divisor,source_rate//divisor).astype(np.complex64)


# Водоспад: спектр запису в часі для відображення. Повертає список рядків (кожен — 512 значень у дБ).
def waterfall(iq, maximum_rows=2400, bins=512):
    rows=[]
    # Приблизно 4 рядки на секунду, але не більше maximum_rows.
    count=min(maximum_rows,max(4,math.ceil(iq.duration*4)))
    for k in range(count):
        # Позиція рядка рівномірно по файлу (з обмеженням, щоб вистачило 4*bins відліків).
        start=min(max(0,iq.frames-4*bins),round(k*iq.frames/count))
        z=iq.read(start,4*bins)
        if len(z)<4*bins:
            z=np.pad(z,(0,4*bins-len(z)))
        # 4 FFT-кадри з вікном Ханна. Усереднення потужності зменшує шум.
        p=np.abs(fft.fftshift(fft.fft(z.reshape(4,bins)*np.hanning(bins),axis=1),axes=1))**2
        db=10*np.log10(np.maximum(p.mean(axis=0),1e-12))
        # Прибираємо DC-пік посередині. Значення округлюємо до 0,1 дБ, щоб JSON був меншим.
        db[bins//2]=(db[bins//2-1]+db[bins//2+1])/2
        rows.append(np.round(db,1).tolist())
    return rows
