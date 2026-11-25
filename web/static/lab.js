'use strict';
// Скорочення для document.getElementById.
const $=id=>document.getElementById(id);
// Форматування чисел за українськими правилами (до 3 знаків після коми).
const nf=new Intl.NumberFormat('uk',{maximumFractionDigits:3});
// 0.934 → «93,9%».
const percent=x=>`${(100*x).toFixed(1).replace('.',',')}%`;
// Підписи класів для таблиці та рамок на водоспаді.
const names={fm:'FM',non_fm:'Не FM'};
// Кольори класів: FM — зелений, не FM — жовтий.
const colors={fm:'#4ce7ba',non_fm:'#f5ba5b'};
// Елементи водоспаду: canvas для малювання, scroller — прокрутка в часі, seek — повзунок часу.
const canvas=$('waterfall'),ctx=canvas.getContext('2d');
const scroller=$('waterfallScroll'),content=$('waterfallContent'),seek=$('timeSeek');
// Ширина лівого поля (px) для підписів часу.
const gutter=56;
// Стан сторінки: вибраний файл, результат аналізу, готове зображення водоспаду,
// ознака «йде аналіз», висота повної прокрутки, id відкладеного перемальовування,
// режим сирого I/Q та ознака «файл ще читається».
let chosen=null,report=null,raster=null,busy=false,fullHeight=480,redraw=0,rawInput=false,readingFile=false;

// Секунди → «хв:сек» (з десятими, якщо вони є).
function formatTime(seconds){
 const tenths=Math.round(Math.max(0,seconds)*10), minutes=Math.floor(tenths/600);
 const rest=((tenths%600)/10).toFixed(1).replace(/\.0$/,'').replace('.',',');
 return `${minutes}:${rest.padStart(rest.includes(',')?4:2,'0')}`;
}
// Рядок стану внизу; error=true підсвічує його як помилку.
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
// Вмикає/вимикає кнопки залежно від стану (під час аналізу вибір файлу заблоковано).
function available(){$('analyze').disabled=busy||!chosen||readingFile;$('file').disabled=busy;}
// Блок висновку: стан (idle/pending/fm/non_fm/unknown), заголовок, пояснення, додаткова статистика.
function verdict(state,title,detail,stats=''){
 $('fileVerdict').dataset.state=state;
 $('verdictTitle').textContent=title;$('verdictDetail').textContent=detail;
 $('verdictStats').textContent=stats;$('verdictStats').hidden=!stats;
}
// Читає JSON-відповідь сервера; при помилці кидає виняток з текстом від сервера.
async function json(response){
 const body=await response.json();
 if(!response.ok)throw Error(typeof body.detail==='string'?body.detail:'Помилка запиту.');
 return body;
}
// Користувач вибрав файл: скидаємо попередній результат і читаємо початок файлу (64 КБ),
// щоб визначити, WAV це чи сирий I/Q, і дістати центральну частоту.
async function select(file){
 chosen=file;report=null;raster=null;
 verdict('idle','Очікує на аналіз','Висновок з’явиться після обробки всього файлу.');
 readingFile=true;available();
 // Центральна частота з назви файлу (напр. ..._96400000Hz_...).
 const center=Number(file.name.match(/(?:^|[_-])(\d{7,10})\s*Hz(?:[_\-.]|$)/i)?.[1])/1e6;
 $('center').value=center||'';
 scroller.hidden=$('results').hidden=$('waterfallTools').hidden=$('timeNavigation').hidden=true;
 $('empty').hidden=false;$('position').textContent='';
 $('rawSettings').hidden=true;
 try{
  // Початок файлу: сигнатура RIFF/WAVE означає заголовковий WAV.
  const header=new Uint8Array(await file.slice(0,65536).arrayBuffer());
  if(chosen!==file)return;
  const magic=String.fromCharCode(...header.slice(0,4));
  const wave=String.fromCharCode(...header.slice(8,12));
  // Деякі різновиди WAV не підтримуються.
  if(['RF64','BW64','RIFX','riff'].includes(magic))throw Error('Цей різновид WAV поки не підтримується. Потрібен PCM WAV або сирий I/Q без заголовка.');
  // Немає RIFF/WAVE — вважаємо файл сирим I/Q і показуємо поля частоти та формату.
  rawInput=magic!=='RIFF'||wave!=='WAVE';
  // Частоти в назві немає: шукаємо її в блоці auxi заголовка WAV (так пише SDR#).
  if(!rawInput&&!center){
   const view=new DataView(header.buffer);
   for(let p=12;p+8<=header.length;){
    const tag=String.fromCharCode(...header.slice(p,p+4)),size=view.getUint32(p+4,true);
    if(tag==='data')break;
    if(tag==='auxi'&&size>=36&&p+44<=header.length){$('center').value=view.getUint32(p+40,true)/1e6;break;}
    p+=8+size+(size%2);
   }
  }
  $('rawSettings').hidden=!rawInput;
  $('sampleRate').value='2.048';$('sampleFormat').value='u8';
  status(`${file.name} · ${(file.size/1e6).toFixed(1)} МБ`);
 }catch(error){if(chosen===file){chosen=null;status(error.message,true);}}
 finally{if(chosen===file||!chosen){readingFile=false;available();}}
}
// Обробники: вибір файлу та перемикач «Лише FM».
$('file').onchange=()=>{if($('file').files[0])select($('file').files[0]);};
$('fmOnly').onchange=()=>{if(report){draw();table();}};

// Готує зображення водоспаду: значення дБ → кольори палітри.
// Межі яскравості беруться з перцентилів (13% і 99,5%), щоб шум не засліплював картинку.
function buildRaster(){
 const rows=report.rows,cols=rows[0].length;
 const values=[],stride=Math.max(1,Math.floor(rows.length/360));
 for(let y=0;y<rows.length;y+=stride)values.push(...rows[y]);
 values.sort((a,b)=>a-b);
 const low=values[Math.floor(.13*values.length)],high=values[Math.floor(.995*values.length)];
 const off=document.createElement('canvas');off.width=cols;off.height=rows.length;
 const pixels=off.getContext('2d').createImageData(cols,rows.length);
 const palette=[[7,23,34],[15,53,84],[19,116,133],[107,209,176],[249,233,147]];
 for(let y=0;y<rows.length;y++)for(let x=0;x<cols;x++){
  const t=Math.max(0,Math.min(.99999,(rows[y][x]-low)/Math.max(1,high-low)))*4;
  const k=Math.floor(t),fraction=t-k,i=4*(y*cols+x);
  for(let c=0;c<3;c++)pixels.data[i+c]=palette[k][c]*(1-fraction)+palette[k+1][c]*fraction;
  pixels.data[i+3]=255;
 }
 off.getContext('2d').putImageData(pixels,0,0);raster=off;
}
// Виявлення, які треба показати (з урахуванням фільтра «Лише FM»).
function visible(){return report.detections.filter(d=>!$('fmOnly').checked||d.class==='fm');}
// Яку частину запису (в секундах) зараз видно на водоспаді.
function viewport(){
 const start=scroller.scrollTop/fullHeight*report.duration_s;
 const seconds=canvas.height/fullHeight*report.duration_s;
 return {start,end:Math.min(report.duration_s,start+seconds),seconds};
}
// Малює водоспад, рамки виявлень, сітку й підписи часу для видимої частини.
function draw(){
 if(!report||!raster||scroller.hidden)return;
 $('position').textContent='';
 const {start,end,seconds}=viewport(),width=canvas.width-gutter;
 ctx.fillStyle='#071722';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.imageSmoothingEnabled=false;
 ctx.drawImage(raster,0,start/report.duration_s*raster.height,raster.width,seconds/report.duration_s*raster.height,gutter,0,width,canvas.height);
 ctx.save();ctx.beginPath();ctx.rect(gutter,0,width,canvas.height);ctx.clip();
 // span — ширина спектра в МГц; minimum — частота лівого краю.
 const span=report.sample_rate_hz/1e6,minimum=report.center_mhz-span/2;
 // Рамка для кожного виявлення: зелена — FM, жовта — не FM. Положення за частотою й часом.
 for(const d of visible()){
  if(d.end_s<start||d.start_s>end)continue;
  const boxWidth=(d.class==='fm'?.19:.075)/span*width;
  const x=gutter+(d.frequency_mhz-minimum)/span*width-boxWidth/2;
  const y=(d.start_s-start)/seconds*canvas.height,height=(d.end_s-d.start_s)/seconds*canvas.height;
  ctx.fillStyle=colors[d.class]+'16';ctx.fillRect(x,y,boxWidth,height);
  ctx.strokeStyle=colors[d.class];ctx.lineWidth=d.class==='fm'?2:1;ctx.strokeRect(x,y,boxWidth,height);
  if(Math.min(canvas.height,y+height)-Math.max(0,y)>=16){
   const labelY=Math.max(0,y);
   ctx.font='12px Arial';ctx.fillStyle='#071722';ctx.fillRect(x+1,labelY+1,Math.min(boxWidth-2,92),16);
   ctx.fillStyle=colors[d.class];ctx.fillText(names[d.class],x+5,labelY+13,Math.max(5,boxWidth-10));
  }
 }
 ctx.restore();
 // Крок сітки часу підбираємо так, щоб на екрані було близько 7 ліній.
 const step=[1,2,5,10,15,30,60,120].find(s=>s>=seconds/7)||120;
 ctx.font='11px Arial';ctx.lineWidth=1;
 for(let t=Math.ceil(start/step)*step;t<=end+.00001;t+=step){
  const y=(t-start)/seconds*canvas.height;
  ctx.strokeStyle='#afc3c425';ctx.beginPath();ctx.moveTo(gutter,y);ctx.lineTo(canvas.width,y);ctx.stroke();
  ctx.fillStyle='#afc3c4';ctx.fillText(formatTime(t),3,Math.max(12,Math.min(canvas.height-4,y+4)));
 }
 seek.value=start;
 $('viewRange').textContent=`${formatTime(start)}–${formatTime(end)}`;
}
// Перемальовуємо не частіше одного разу на кадр (економимо ресурси при прокрутці).
function queueDraw(){
 if(redraw)return;
 redraw=requestAnimationFrame(()=>{redraw=0;draw();});
}
// Перераховує розміри водоспаду при зміні масштабу часу чи розміру вікна, зберігаючи поточну позицію.
function layout(){
 if(!report||scroller.hidden)return;
 const oldStart=scroller.scrollTop/fullHeight*report.duration_s;
 const height=scroller.clientHeight;
 const choice=$('timeScale').value;
 const seconds=choice==='all'?report.duration_s:Math.min(report.duration_s,Number(choice));
 fullHeight=Math.max(height,Math.ceil(report.duration_s/seconds*height));
 content.style.height=`${fullHeight}px`;
 canvas.width=1024;canvas.height=height;canvas.style.height=`${height}px`;
 seek.max=Math.max(0,report.duration_s-height/fullHeight*report.duration_s);
 seek.disabled=fullHeight===height;
 scroller.scrollTop=oldStart/report.duration_s*fullHeight;draw();
}
// Прокручує водоспад до моменту time (с).
function jump(time){
 scroller.scrollTop=Math.max(0,Math.min(time,Number(seek.max)))/report.duration_s*fullHeight;
 draw();
}
// Підписки на прокрутку, зміну масштабу, повзунок і зміну розміру.
scroller.onscroll=queueDraw;
$('timeScale').onchange=layout;
seek.oninput=()=>jump(Number(seek.value));
new ResizeObserver(()=>{if(report)layout();}).observe(scroller);

// Заповнює таблицю виявлень: клас, частота, часовий проміжок (кнопка переходу) і впевненість.
function table(){
 $('detections').replaceChildren();
 for(const d of visible()){
  const row=$('detections').insertRow();row.className=d.class;
  row.insertCell().textContent=names[d.class];
  row.insertCell().textContent=`${nf.format(d.frequency_mhz)} МГц`;
  const jumpButton=document.createElement('button');jumpButton.className='jump-time';
  jumpButton.textContent=`${formatTime(d.start_s)}–${formatTime(d.end_s)}`;
  jumpButton.title='Перейти до ділянки на водоспаді';
  jumpButton.onclick=()=>{jump(d.start_s);scroller.scrollIntoView({block:'nearest'});};
  row.insertCell().append(jumpButton);row.insertCell().textContent=percent(d.score);
 }
 $('summary').textContent=`FM · ${report.detections.filter(d=>d.class==='fm').length} · Не FM · ${report.detections.filter(d=>d.class==='non_fm').length}`;
 if(!visible().length){const cell=$('detections').insertRow().insertCell();cell.colSpan=4;cell.textContent='Впевнених розпізнавань немає.';}
}
// Під курсором показуємо частоту й час у відповідній точці водоспаду.
canvas.onmousemove=e=>{
 if(!report)return;
 const bounds=canvas.getBoundingClientRect(),{start,seconds}=viewport();
 const plotStart=gutter/canvas.width*bounds.width;
 const x=Math.max(0,Math.min(1,(e.clientX-bounds.left-plotStart)/(bounds.width-plotStart)));
 const y=(e.clientY-bounds.top)/bounds.height;
 $('position').textContent=`${nf.format(report.center_mhz+(x-.5)*report.sample_rate_hz/1e6)} МГц · ${formatTime(start+y*seconds)}`;
};
// Показує результат аналізу: висновок, підписи частот під водоспадом, таблицю.
function show(result){
 report=result;buildRaster();table();
 const conclusion=result.summary;
 verdict(conclusion.state,conclusion.title,conclusion.detail,
   conclusion.fm_regions?`Час із FM: ${formatTime(conclusion.fm_time_s)} із ${formatTime(result.duration_s)}. Ділянок FM: ${conclusion.fm_regions}.`:
   conclusion.non_fm_regions?`Ділянок не FM: ${conclusion.non_fm_regions}.`:'');
 const half=result.sample_rate_hz/2e6;
 for(const [id,value] of [['left',result.center_mhz-half],['middle',result.center_mhz],['right',result.center_mhz+half]])$(id).textContent=`${nf.format(value)} МГц`;
 $('fileInfo').textContent=`${result.filename} · ${formatTime(result.duration_s)}`;
 $('totalTime').textContent=formatTime(result.duration_s);
 $('empty').hidden=true;
 scroller.hidden=$('results').hidden=$('waterfallTools').hidden=$('timeNavigation').hidden=false;
 scroller.scrollTop=0;fullHeight=scroller.clientHeight;layout();
}
// Надсилає файл на сервер (POST /api/analyze) з індикатором завантаження. Повертає {id} задачі.
function upload(){
 return new Promise((resolve,reject)=>{
  const xhr=new XMLHttpRequest(),q=new URLSearchParams({name:chosen.name});
  if($('center').value)q.set('center_mhz',$('center').value);
  // Для сирого I/Q передаємо ще частоту відліків і формат.
  if(rawInput){
   const rate=Number($('sampleRate').value);
   if(!Number.isFinite(rate)||rate<.32||rate>5)throw Error('Укажи частоту відліків від 0,32 до 5 МГц.');
   q.set('raw_rate_hz',String(Math.round(rate*1e6)));
   q.set('raw_format',$('sampleFormat').value);
  }
  xhr.open('POST',`/api/analyze?${q}`);xhr.responseType='json';xhr.setRequestHeader('Content-Type','application/octet-stream');
  xhr.upload.onprogress=e=>{if(e.lengthComputable){$('progress').value=100*e.loaded/e.total;status(`Передаю запис · ${Math.round(100*e.loaded/e.total)}%`);}};
  xhr.onload=()=>xhr.status===202?resolve(xhr.response):reject(Error(typeof xhr.response?.detail==='string'?xhr.response.detail:'Не вдалося завантажити запис.'));
  xhr.onerror=()=>reject(Error('З’єднання із сервером перервано.'));xhr.send(chosen);
 });
}
// Запускає аналіз: завантаження файлу, потім опитування стану задачі кожні 0,6 с, потім отримання результату.
async function analyze(){
 if(!chosen||busy)return;
 verdict('pending','Аналізую весь запис','Підсумок з’явиться після завершення обробки.');
 busy=true;available();$('center').disabled=$('sampleRate').disabled=$('sampleFormat').disabled=true;$('progress').hidden=false;$('progress').value=0;
 try{
  const {id}=await upload();
  for(;;){
   const job=await json(await fetch(`/api/jobs/${id}`));
   $('progress').value=job.progress;
   if(job.state==='failed')throw Error(job.message);
   status(job.state==='queued'?'Очікує на обробку…':`Аналізую запис · ${job.progress}%`);
   if(job.state==='ready'){show(await json(await fetch(`/api/jobs/${id}/result`)));status('Аналіз готовий');break;}
   await new Promise(resolve=>setTimeout(resolve,600));
  }
 }catch(error){status(error.message,true);verdict('unknown','Аналіз не завершено','Перевір файл і параметри запису.');}finally{busy=false;available();$('center').disabled=$('sampleRate').disabled=$('sampleFormat').disabled=false;$('progress').hidden=true;}
}
// Кнопка «Аналізувати».
$('analyze').onclick=analyze;
