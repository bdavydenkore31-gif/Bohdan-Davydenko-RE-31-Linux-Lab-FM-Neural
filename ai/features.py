"""Входи нейромережі, сформовані безпосередньо з I/Q."""
import numpy as np
from scipy import fft

# Частота відліків каналу (Гц). Усі вікна зводяться до 320 кГц.
RATE=320000
# Довжина одного вікна для моделі: 8192 відліки ≈ 25,6 мс.
SAMPLES=8192
# Розмір FFT для спектра. 8192 / 256 = 32 кадри на вікно.
NFFT=256
# Сітка частот (96 точок від −150 до +150 кГц), на яку зводиться спектр, щоб вхід моделі мав сталий розмір.
GRID=np.linspace(-150000,150000,96)
# Назви класів: FM і все інше.
CLASSES=("fm","non_fm")


# Спектральні ознаки: усереднена за часом спектрограма вікна + мінливість спектра в часі.
# Разом 4×96 + 96 = 480 чисел.
def features(z):
    z=np.asarray(z,dtype=np.complex64)
    # Коротке вікно доповнюємо нулями, довге обрізаємо до SAMPLES.
    if len(z)<SAMPLES:
        z=np.pad(z,(0,SAMPLES-len(z)))
    z=z[:SAMPLES]
    # Ріжемо вікно на 32 кадри по 256 відліків.
    frames=z.reshape(-1,NFFT)
    # Спектр кожного кадру: вікно Ханна зменшує розтікання, fftshift ставить 0 Гц у центр. Потужність = |X|².
    power=np.abs(fft.fftshift(fft.fft(frames*np.hanning(NFFT),axis=1),axes=1))**2
    # Переводимо в децибели (нижня межа не дає log(0)).
    db=10*np.log10(np.maximum(power,1e-12))
    # Центральна точка (0 Гц) забруднена DC-зсувом приймача. Замінюємо середнім від сусідів.
    db[:,NFFT//2]=(db[:,NFFT//2-1]+db[:,NFFT//2+1])/2
    # Вісь частот у Гц, що відповідає стовпцям спектра.
    axis=fft.fftshift(fft.fftfreq(NFFT,1/RATE))
    # Чотири часові зрізи передають зміну форми спектра.
    pooled=db.reshape(4,8,NFFT).mean(axis=1)
    # Інтерполюємо кожен часовий зріз на фіксовану сітку GRID.
    spectrum=np.array([np.interp(GRID,axis,row) for row in pooled])
    # Віднімаємо медіану (рівень шуму/підсилення) й масштабуємо, щоб значення були порядку 1.
    spectrum=(spectrum-np.median(spectrum))/12
    # Стандартне відхилення в часі для кожної частоти: у FM спектр «дихає», у чистої несучої майже ні.
    variation=np.interp(GRID,axis,np.std(db,axis=0))/12
    # Склеюємо в один вектор. clip обмежує викиди на рівні ±5.
    return np.concatenate((np.clip(spectrum.ravel(),-5,5),variation)).astype(np.float32)


def hybrid_features(z):
    """Спектр разом із часовою структурою амплітуди й фази I/Q."""
    z=np.asarray(z,dtype=np.complex64)[:SAMPLES]
    if len(z)<SAMPLES:
        z=np.pad(z,(0,SAMPLES-len(z)))
    z=z/(np.sqrt(np.mean(np.abs(z)**2))+1e-12)
    # Ознаки амплітуди: у FM обвідна майже постійна, у AM та ADS-B — ні.
    amplitude=np.abs(z)
    power=amplitude**2
    # Нормалізація робить ознаки незалежними від підсилення приймача.
    extra=[np.mean(amplitude),np.std(amplitude),np.mean(power**2),
           np.mean(power**3),np.mean(power**4)]
    extra.extend(np.quantile(amplitude,[.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    # Автокореляція обвідної на різних зсувах (lag) показує повторювану структуру: імпульси, символи.
    centered=amplitude-amplitude.mean()
    variance=np.mean(centered**2)+1e-8
    for lag in (1,2,4,8,16,32,64,128):
        extra.append(np.mean(centered[lag:]*centered[:-lag])/variance)
    # Розкид потужності, усередненої по блоках різної довжини: відрізняє імпульсні сигнали від безперервних.
    for factor in (2,8,32,128):
        levels=power.reshape(-1,factor).mean(axis=1)
        extra.extend((np.std(levels),np.quantile(levels,.95)-np.quantile(levels,.05)))
    # Різниці фаз зберігають модуляцію; постійна фаза та зсув несучої не потрібні.
    phase=np.angle(z[1:]*z[:-1].conj())
    unit=np.exp(1j*phase)
    phase=np.angle(unit*np.exp(-1j*np.angle(unit.mean())))
    extra.extend(np.quantile(np.abs(phase),[.1,.25,.5,.75,.9,.95,.99]))
    # Кругові моменти нормованої автокореляції фази: чутливі до типу модуляції (FM, PSK тощо).
    for lag in (1,2,4,8,16,32):
        product=z[lag:]*z[:-lag].conj()
        magnitude=np.abs(product)
        circular=np.divide(product,magnitude,out=np.ones_like(product),where=magnitude>0)
        # Степені одиничного комплексного числа замінюють повторні exp/sin/cos.
        second=circular*circular
        extra.extend((abs(circular.mean()),abs(second.mean()),abs((second*second).mean())))
        change=circular[1:]*circular[:-1].conj()
        extra.extend((change.real.mean(),(change*change).real.mean()))
    # Спектр миттєвої частоти (фази): у FM тут видно звукову смугу та пілот 19 кГц стерео. Згортаємо у 16 смуг.
    phase_power=np.abs(fft.rfft(phase[:7936].reshape(31,256)*np.hanning(256),axis=1))**2
    phase_power=phase_power.mean(axis=0)[1:]
    phase_power/=phase_power.sum()+1e-12
    extra.extend(np.log1p(1000*phase_power.reshape(16,8).sum(axis=1)))
    # Логарифм стримує великий динамічний діапазон імпульсних сигналів ADS-B.
    extra=np.sign(extra)*np.log1p(np.abs(extra))
    # Підсумковий вектор: спектральні ознаки + додаткові статистики.
    return np.concatenate((features(z),extra)).astype(np.float32)


# Реєстр способів формування ознак. Ім'я береться з training.json (поле feature_kind).
EXTRACTORS={'hybrid':hybrid_features}
