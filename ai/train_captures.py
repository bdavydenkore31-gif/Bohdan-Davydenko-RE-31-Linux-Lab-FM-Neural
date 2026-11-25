"""Навчання на реальних розмічених I/Q та контрольованих перетвореннях."""
import copy
import json
import time

import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ai.augment import NEGATIVES,augment
from ai.features import EXTRACTORS


# Читає розмічений набір і перевіряє, що часові частини різних вибірок не перетинаються в межах кожного джерела.
# Зберігає навчену мережу в .npz: параметри нормалізації, класи, ваги й зсуви кожного шару.
# Спершу пишемо в тимчасовий файл, потім підміняємо, щоб при збої не лишився пошкоджений classifier.npz.
def save_model(path,net,scaler):
    values={'mean':scaler.mean_.astype(np.float32),'scale':scaler.scale_.astype(np.float32),'classes':net.classes_}
    for index,(weights,bias) in enumerate(zip(net.coefs_,net.intercepts_)):
        values[f'w{index}']=weights.astype(np.float32)
        values[f'b{index}']=bias.astype(np.float32)
    pending=path.with_suffix('.tmp.npz')
    np.savez_compressed(pending,**values)
    pending.replace(path)


def read_bank(path):
    with np.load(path,allow_pickle=False) as bank:
        result={key:bank[key] for key in bank.files}
    iq=result.pop('iq').astype(np.float32)/32768
    result['z']=iq[:,:,0]+1j*iq[:,:,1]
    for source in np.unique(result['source']):
        for split in (0,1):
            left=(result['source']==source)&(result['split']==split)
            right=(result['source']==source)&(result['split']==split+1)
            if result['end_s'][left].max()>=result['start_s'][right].min():
                raise ValueError('Перетин навчання та перевірки.')
    return result


# Формує count прикладів для вибірки split (0/1/2) і повертає (ознаки, мітки, тип).
# Склад: 50% FM. Серед негативів ~40% синтетичні (з FM через augment), решта — реальні не-FM записи.
def make_set(bank,split,count,seed,representation):
    rng=np.random.default_rng(seed)
    extractor=EXTRACTORS[representation]
    indices=np.flatnonzero(bank['split']==split)
    positives=indices[bank['label'][indices]=='fm']
    groups={}
    for i in indices:
        if bank['label'][i]=='non_fm':
            groups.setdefault(str(bank['source'][i]),[]).append(i)
    positive_groups={str(source):positives[bank['source'][positives]==source]
                     for source in np.unique(bank['source'][positives])}
    rows=[];labels=[];kinds=[]
    # Кожен запис має однакову вагу незалежно від кількості його каналів.
    for k in range(count):
        # Парні приклади — позитив (FM), непарні — негатив.
        positive=k%2==0
        synthetic=not positive and rng.random()<.4
        # FM та синтетичні негативи роблять із реальних FM-вікон. Реальні негативи беруть із їхніх груп.
        if positive or synthetic:
            group=positive_groups[rng.choice(list(positive_groups))]
        else:
            group=groups[rng.choice(list(groups))]
        # Випадкове вікно з обраної групи.
        index=int(rng.choice(group));z=bank['z'][index].copy()
        kind=str(bank['kind'][index])
        # Синтетичний негатив: перетворюємо FM на AM/CW/QPSK/…, інакше (65%) просто додаємо реалістичні спотворення.
        if synthetic:
            kind=str(rng.choice(NEGATIVES));z,_=augment(z,kind,rng)
        elif rng.random()<.65:
            # Перетворення fm зберігає вхідну модуляцію і для негативних записів.
            z,_=augment(z,'fm',rng)
        rows.append(extractor(z));labels.append('fm' if positive else 'non_fm');kinds.append(kind)
        if (k+1)%1000==0:
            print(f'  ознаки {k+1} / {count}',flush=True)
    return np.asarray(rows),np.asarray(labels),np.asarray(kinds)


# Метрики за порогом: точність, recall FM, частка хибних FM, матриця помилок і частка «FM» за кожним типом джерела.
def metrics(truth,probabilities,threshold,kinds):
    truth=np.asarray(truth)=='fm';predicted=probabilities>=threshold
    return {'samples':len(truth),'accuracy':float(np.mean(predicted==truth)),
            'fm_recall':float(np.mean(predicted[truth])) if truth.any() else None,
            'false_fm_rate':float(np.mean(predicted[~truth])) if (~truth).any() else None,
            'confusion_matrix_non_fm_fm':confusion_matrix(truth,predicted,labels=[False,True]).tolist(),
            'by_kind':{str(kind):{'samples':int(np.sum(kinds==kind)),
                                 'fm_fraction':float(np.mean(predicted[kinds==kind]))}
                       for kind in np.unique(kinds)}}


# Навчання на розмічених даних: MLP → калібрування Платта → вибір порога на validation →
# оцінка на test та на реальних (неаугментованих) test-вікнах → експорт.
def fit(path,count,epochs,dest,representation='hybrid'):
    began=time.monotonic();bank=read_bank(path)
    sets=[]
    for split,n in enumerate((count,count//4,count//4)):
        print(f'Готую {("train","validation","test")[split]}: {n} прикладів',flush=True)
        sets.append(make_set(bank,split,n,4919+split*1217,representation))
    (xt,yt,_),(xv,yv,kv),(xe,ye,ke)=sets
    # Стандартизація (параметри лише з train).
    scaler=StandardScaler().fit(xt)
    xt,xv,xe=(scaler.transform(x).astype(np.float32) for x in (xt,xv,xe))
    # Сильніша L2-регуляризація (alpha=.05) і менший крок, ніж у train.py: менше перенавчання.
    net=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',
                      alpha=.05,batch_size=128,learning_rate_init=.0007,random_state=2741)
    # Епохи: вибираємо найкращу модель за log-loss на validation (стабільніше за точність).
    # Рання зупинка після 15 епох без покращення.
    best=None;best_loss=float('inf');history=[];stale=0
    with threadpool_limits(limits=2):
        for epoch in range(1,epochs+1):
            net.partial_fit(xt,yt,classes=np.array(['fm','non_fm']))
            p=net.predict_proba(xv)[:,0]
            truth=yv=='fm';clipped=np.clip(p,1e-6,1-1e-6)
            loss=float(-np.mean(truth*np.log(clipped)+(1-truth)*np.log(1-clipped)))
            score=float(np.mean((p>=.5)==truth))
            history.append({'epoch':epoch,'validation_log_loss':loss,'validation_accuracy':score})
            print(f'Епоха {epoch:02d}: val accuracy {score:.4f}, log loss {loss:.4f}',flush=True)
            if loss<best_loss-.0001:
                best=copy.deepcopy(net);best_loss=loss;stale=0
            else:stale+=1
            if stale>=15:break
        # Сирі ймовірності FM на validation.
        val_raw=best.predict_proba(xv)[:,0]
        # Частини валідаційних прикладів для калібрування та вибору порога.
        calibration=LogisticRegression(C=1,random_state=381)
        calibrate=np.arange(len(yv))%4<2
        calibration.fit(logit(np.clip(val_raw[calibrate],1e-7,1-1e-7)).reshape(-1,1),yv[calibrate]=='fm')
        # Параметри калібрування. calibrated() застосовує їх до будь-яких ймовірностей.
        slope=float(calibration.coef_[0,0]);intercept=float(calibration.intercept_[0])
        calibrated=lambda p:expit(slope*logit(np.clip(p,1e-7,1-1e-7))+intercept)
        val=calibrated(val_raw)
        choose=~calibrate;vtruth=yv=='fm'
        # Поріг обирається лише на validation, з обмеженням хибних FM у кожній групі.
        thresholds=np.unique(np.r_[.8,np.linspace(.8,.9995,200),val[choose]])
        threshold=1.0
        for t in thresholds[thresholds>=.8]:
            if all(np.mean(val[choose&(kv==kind)]>=t)<=.01
                   for kind in np.unique(kv[choose&~vtruth])):
                threshold=float(t);break
        # Ймовірності на test (штучні приклади).
        test=calibrated(best.predict_proba(xe)[:,0])
    # Неперетворені тестові вікна показують якість без впливу аугментації.
    real_index=np.flatnonzero(bank['split']==2)
    xr=np.asarray([EXTRACTORS[representation](bank['z'][i]) for i in real_index])
    with threadpool_limits(limits=2):
        real=calibrated(best.predict_proba(scaler.transform(xr))[:,0])
    # Звіт для training.json: параметри, пороги, калібрування, метрики.
    result={'schema':3,'feature_kind':representation,'architecture':[xt.shape[1],128,64,1],
            'framework':'scikit-learn MLPClassifier','threshold':threshold,'non_fm_threshold':.8,
            'calibration':{'slope':slope,'intercept':intercept,'method':'Platt scaling, validation only'},
            'inference_windows':3,'train_samples':len(yt),'validation_samples':len(yv),
            'best_validation_log_loss':best_loss,'history':history,
            'test':metrics(ye,test,threshold,ke),
            'real_test':metrics(bank['label'][real_index],real,threshold,bank['kind'][real_index]),
            'test_roc_auc':float(roc_auc_score(ye=='fm',test)),
            'source':json.loads(path.with_suffix('.json').read_text(encoding='utf-8')),
            'scope':'Відкладені часові ділянки цих записів. Це не гарантія для будь-яких модуляцій і приймачів.',
            'elapsed_s':round(time.monotonic()-began,2)}
    dest.mkdir(parents=True,exist_ok=True)
    # Зберігаємо ваги та звіт.
    save_model(dest/'classifier.npz',best,scaler)
    result['weights_bytes']=(dest/'classifier.npz').stat().st_size
    (dest/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # Власний числовий рушій має відтворювати sklearn разом із калібруванням.
    from ai.model import Classifier
    exported=Classifier(dest)
    windows=bank['z'][real_index[:32]]
    with threadpool_limits(limits=2):
        exported.predict(windows)
        start=time.perf_counter();actual=exported.predict(windows)[:,exported.classes.index('fm')]
        result['inference_ms_per_window']=(time.perf_counter()-start)*1000/len(windows)
    # Захист: відповіді власного рушія мають збігатися з sklearn із точністю 2e-5.
    if not np.allclose(actual,real[:len(windows)],atol=2e-5):
        raise RuntimeError('Експортована модель не відтворює sklearn.')
    (dest/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'threshold':threshold,'test':result['test'],'real_test':result['real_test']},ensure_ascii=False,indent=2),flush=True)
    return result
