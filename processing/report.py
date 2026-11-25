"""Підсумок усіх розпізнаних ділянок запису."""


# Скільки секунд запису покрито виявленнями класу kind (перекриті проміжки не рахуються двічі).
def covered_seconds(detections,kind,duration):
    # Одночасний FM на кількох частотах рахуємо як один проміжок часу.
    intervals=sorted((max(0,item['start_s']),min(duration,item['end_s']))
                     for item in detections if item['class']==kind)
    # Об'єднуємо відсортовані інтервали. right — найправіша вже врахована межа.
    total=0.0;right=0.0
    for begin,end in intervals:
        if end>max(begin,right):
            total+=end-max(begin,right)
        right=max(right,end)
    return round(total,3)


# Формує підсумок для інтерфейсу: стан (fm / non_fm / unknown), заголовок, пояснення та статистику.
def summarize(detections,duration):
    fm=sum(item['class']=='fm' for item in detections)
    other=sum(item['class']=='non_fm' for item in detections)
    # Пріоритет: хоч одна FM-ділянка → «у файлі є FM»; далі «не FM»; інакше «недостатньо даних».
    if fm:
        state='fm';title='У файлі є FM'
        detail='Також виявлено сигнали не FM.' if other else 'Виявлено ділянки FM-сигналу.'
    elif other:
        state='non_fm';title='FM не виявлено'
        detail='Розпізнані ділянки належать до класу «не FM».'
    else:
        state='unknown';title='Недостатньо даних для висновку'
        detail='Упевнених розпізнавань FM або не FM немає.'
    return {'state':state,'title':title,'detail':detail,
            'fm_regions':fm,'non_fm_regions':other,
            'fm_time_s':covered_seconds(detections,'fm',duration),
            'non_fm_time_s':covered_seconds(detections,'non_fm',duration)}
