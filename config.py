"""Шляхи проєкту, незалежні від поточної робочої папки."""
from pathlib import Path

# Коренева папка проєкту. Рахується від розташування цього файлу, тому шляхи працюють з будь-якої робочої папки.
ROOT=Path(__file__).resolve().parent
# Тимчасові файли: сюди веб-інтерфейс складає завантажені записи на час аналізу.
DATA_DIR=ROOT/'data'
# Навчена модель (classifier.npz) та її метадані (training.json, evaluation.json).
MODEL_DIR=ROOT/'ai/models'
# Початковий набір сирих I/Q-вікон каналу FM (база для розміченого набору).
DATASET_PATH=ROOT/'ai/datasets/raw_iq_windows.npz'
# Фронтенд: index.html, lab.js, style.css.
STATIC_DIR=ROOT/'web/static'
