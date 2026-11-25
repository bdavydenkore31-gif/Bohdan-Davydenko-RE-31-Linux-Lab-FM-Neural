"""Команда навчання FM-класифікатора на розміченому наборі I/Q."""
import argparse
from pathlib import Path

from ai.train_captures import fit
from config import DATASET_PATH, MODEL_DIR

if __name__=='__main__':
    # Параметри: розмічений набір, кількість навчальних прикладів, максимум епох, куди зберегти модель.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATASET_PATH.with_name('labeled_iq.npz'))
    parser.add_argument('--samples',type=int,default=8000)
    parser.add_argument('--epochs',type=int,default=70)
    parser.add_argument('--output',type=Path,default=MODEL_DIR)
    args=parser.parse_args()
    if args.samples<1000 or args.epochs<1:
        parser.error('Потрібні щонайменше 1000 прикладів і одна епоха.')
    # Усе навчання (дані, MLP, калібрування, поріг, експорт) відбувається в train_captures.fit.
    fit(args.dataset,args.samples,args.epochs,args.output)
