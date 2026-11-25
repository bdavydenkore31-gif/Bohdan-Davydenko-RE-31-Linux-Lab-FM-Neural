"""Запуск веблабораторної розпізнавання FM."""
import argparse

from web.api import create_app

# Створюємо ASGI-застосунок на рівні модуля, щоб його можна було запускати і через `uvicorn app:app`.
app=create_app()

# Цей блок виконується лише при прямому запуску `python app.py`, а не при імпорті.
if __name__=='__main__':
    import uvicorn
    # Параметри командного рядка: адреса й порт, на яких слухатиме веб-сервер.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8010)
    args=parser.parse_args()
    # Запускаємо сервер. За замовчуванням лабораторія відкривається на http://127.0.0.1:8010
    uvicorn.run(app,host=args.host,port=args.port)
