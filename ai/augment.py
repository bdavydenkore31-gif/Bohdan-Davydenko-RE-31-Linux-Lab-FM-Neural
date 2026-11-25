"""Зміни сирих I/Q-вікон для прикладів класу «не FM»."""
import numpy as np
from scipy import fft, signal

from ai.features import RATE, SAMPLES

# Типи перетворень, з яких роблять приклади класу «не FM»:
# AM, незмодульована несуча (CW), цифрова QPSK, сигнал зі зруйнованою фазою та чистий шум.
NEGATIVES=('am','cw','qpsk','phase_scramble','noise')


# Перетворює реальне I/Q-вікно в сигнал заданого типу (kind).
# Для 'fm' сигнал лишається як є. Решта типів — навмисні «не FM» негативи з того самого запису.
def transform_iq(source, kind, rng):
    # Копіюємо (щоб не псувати вихідний масив) і нормуємо потужність до 1: усі типи стають порівнянними за рівнем.
    z=np.array(source,dtype=np.complex64,copy=True)
    z/=np.sqrt(np.mean(np.abs(z)**2))+1e-9
    # FM — це справжній запис без змін.
    if kind=='fm':
        return z
    if kind=='am':
        # Повідомленням є згладжений I-компонент сирого сигналу.
        # ФНЧ на 10 кГц виділяє «звукову» складову з I-каналу. Вона стане повідомленням для AM.
        taps=signal.firwin(65,10000,fs=RATE)
        message=signal.fftconvolve(z.real,taps,mode='same')
        message/=np.std(message)+1e-8
        # AM: амплітуда змінюється за повідомленням (глибина 60–95%), tanh обмежує викиди.
        return (1+rng.uniform(.6,.95)*np.tanh(message)).astype(np.complex64)
    # CW: постійна амплітуда й фаза, тобто несуча без модуляції.
    if kind=='cw':
        return np.ones(len(z),dtype=np.complex64)
    if kind=='qpsk':
        # QPSK: беремо знаки I та Q кожного period-го відліку як «біти» і розтягуємо назад.
        # Виходить цифрова модуляція з випадковою швидкістю символів.
        period=int(rng.integers(3,12))
        bits=np.sign(z.real[::period])+1j*np.sign(z.imag[::period])
        return np.repeat(bits,period)[:len(z)].astype(np.complex64)/np.sqrt(2)
    if kind=='phase_scramble':
        # Амплітудний спектр цілого вікна збережено, фази зруйновано.
        spectrum=fft.fft(z)
        return fft.ifft(np.abs(spectrum)*np.exp(1j*rng.uniform(-np.pi,np.pi,len(z)))).astype(np.complex64)
    # Шум: комплексний гаусів. У половині випадків обмежуємо смугу (спад 10-го порядку в частотній області), як шум приймача в каналі.
    if kind=='noise':
        noise=(rng.normal(size=len(z))+1j*rng.normal(size=len(z))).astype(np.complex64)
        if rng.random()<.5:
            axis=np.abs(fft.fftfreq(len(z),1/RATE))
            noise=fft.ifft(fft.fft(noise)/np.sqrt(1+(axis/rng.uniform(40000,125000))**10))
        return noise/(np.sqrt(np.mean(np.abs(noise)**2))+1e-9)
    # Невідомий тип — це помилка програміста, а не даних.
    raise ValueError(f'Невідоме перетворення I/Q {kind}')


# Аугментація: робить із вікна навчальний приклад, імітуючи спотворення реального приймача
# (зсув частоти, відлуння, шум, зміна рівня). Повертає (вікно, SNR у дБ).
# SNR потрібен, щоб окремо оцінювати якість на слабких сигналах.
def augment(source, kind, rng):
    # Спочатку застосовуємо потрібне перетворення, потім обрізаємо до довжини вікна моделі.
    z=transform_iq(source,kind,rng)
    z=z[:SAMPLES]
    # Випадковий зсув несучої ±9 кГц і випадкова фаза: модель не повинна залежати від точного налаштування частоти.
    t=np.arange(len(z))/RATE
    z*=np.exp(1j*(2*np.pi*rng.uniform(-9000,9000)*t+rng.uniform(-np.pi,np.pi))).astype(np.complex64)
    # У 30% випадків додаємо слабке запізніле відлуння (імітація багатопроменевості).
    if rng.random()<.3:
        z+=rng.uniform(.02,.18)*np.roll(z,int(rng.integers(1,20)))*np.exp(1j*rng.uniform(0,6.28))
    # Невеликий дисбаланс I/Q (±15%) — типова вада реальних приймачів.
    z.real*=rng.uniform(.85,1.15)
    # Знову нормуємо потужність, щоб далі точно задати SNR.
    z/=np.sqrt(np.mean(np.abs(z)**2))+1e-9
    # Додаємо білий шум так, щоб SNR був випадковим: від −2 до 25 дБ.
    snr=rng.uniform(-2,25)
    noise=(rng.normal(size=len(z))+1j*rng.normal(size=len(z))).astype(np.complex64)/np.sqrt(2)
    z+=noise*10**(-snr/20)
    # У 20% випадків додаємо постійну складову (DC-зсув), як у багатьох SDR.
    if rng.random()<.2:
        z+=rng.uniform(.01,.2)*np.exp(1j*rng.uniform(0,6.28))
    # Випадкове підсилення: модель має працювати за будь-якого рівня сигналу.
    z*=rng.uniform(.2,1.5)
    return z.astype(np.complex64),snr
