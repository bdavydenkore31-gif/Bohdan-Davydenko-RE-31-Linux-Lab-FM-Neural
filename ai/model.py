"""Відтворення навченої MLP з числових ваг без pickle."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from ai.features import EXTRACTORS


# Легкий власний інференс MLP: завантажує ваги з .npz і рахує прямий прохід на numpy,
# без sklearn і pickle (безпечніше й швидше стартує).
class Classifier:
    def __init__(self, folder):
        folder=Path(folder)
        # Метадані навчання: тип ознак, пороги, калібрування.
        self.training=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        # allow_pickle=False: завантажуємо лише числові масиви, без виконання чужого коду.
        with np.load(folder/'classifier.npz',allow_pickle=False) as weights:
            # mean/scale — параметри стандартизації ознак; layers — пари (ваги, зсуви) шарів мережі.
            self.mean=weights['mean']
            self.scale=weights['scale']
            self.classes=weights['classes'].tolist()
            self.layers=[(weights[f'w{i}'],weights[f'b{i}']) for i in range(3)]

    # Повертає матрицю ймовірностей [кількість вікон × кількість класів] для списку I/Q-вікон.
    def predict(self, samples):
        # Беремо той самий спосіб ознак, що був при навчанні, і стандартизуємо їх.
        extractor=EXTRACTORS[self.training['feature_kind']]
        x=(np.array([extractor(z) for z in samples])-self.mean)/self.scale
        # Приховані шари: лінійне перетворення + ReLU.
        for w,b in self.layers[:-1]:
            x=np.maximum(0,x@w+b)
        # Вихідний шар дає логіти.
        w,b=self.layers[-1]
        logits=x@w+b
        # Один вихід (сигмоїда): p — ймовірність другого класу, перша колонка — доповнення до 1.
        p=expit(logits[:,0])
        probabilities=np.column_stack((1-p,p))
        # Калібрування Платта: підганяє ймовірності так, щоб «0,9» справді означало ~90% упевненості.
        # Параметри знайдено на validation.
        calibration=self.training.get('calibration')
        if calibration:
            index=self.classes.index('fm')
            raw=probabilities[:,index]
            odds=np.log(np.maximum(raw,1e-7)/np.maximum(1-raw,1e-7))
            calibrated=expit(calibration['slope']*odds+calibration['intercept'])
            probabilities[:,index]=calibrated
            probabilities[:,1-index]=1-calibrated
        return probabilities
