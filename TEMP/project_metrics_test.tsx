import React, { useMemo, useState, useEffect } from "react";
import { Download, ChevronLeft } from "lucide-react";

// =====================================
// SynapSight — Страница метрик (самодостаточная)
// Причина правки: сборка падала из-за экспорта из './index.tsx', которого нет.
// Теперь файл полностью автономный: без внешних импортов локальных файлов.
// Возвращены: плавные линии, анимация появления и ховер-подсказка (секунда + значение).
// =====================================

export default function MetricsPage(){
  runDevTests();

  const video = { name:'Brand_15sec.mp4', createdISO:new Date().toISOString(), duration:'00:15', id:'demo1' };
  const seconds = durationToSeconds(video.duration) || 60;
  const scores = scoreFromId(video.id, seconds);
  const similar = similarBrands(video.name);
  const recs = generateAIRecommendations(scores);

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-clip">
      <StyleBrand />

      <main className="mx-auto max-w-7xl px-5 py-10 space-y-8">
        {/* Кнопка назад вне заголовка */}
        <div className="flex justify-start mb-4">
          <button onClick={()=>window.history.back()} className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/15 hover:border-white/30 text-sm">
            <ChevronLeft className="w-4 h-4"/> вернуться в личный кабинет
          </button>
        </div>

        {/* Шапка со сводкой */}
        <header className="card rounded-2xl p-6 space-y-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-4 min-w-0">
              <div className="w-10 h-7 rounded bg-white/10 flex items-center justify-center text-[10px] opacity-80">MP4</div>
              <div className="min-w-0">
                <p className="text-sm opacity-80">Метрики ролика</p>
                <h1 className="text-2xl font-bold truncate">{video.name} • загружен {new Date(video.createdISO).toLocaleString('ru-RU')}</h1>
              </div>
            </div>
            <button className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/15 hover:border-white/30 text-sm"><Download className="w-4 h-4"/> Экспорт PDF</button>
          </div>

          {/* Мини-метрики во всю ширину блока заголовка */}
          <div className="grid sm:grid-cols-3 gap-4">
            <SmallStat label="Внимание" value={scores.attention} caption="доля времени с удержанным взглядом" />
            <SmallStat label="Вовлечённость" value={scores.engagement} caption="поведенческий интерес и интеракции" />
            <SmallStat label="Запоминаемость" value={scores.memory} caption="вероятность воспоминания через 24ч" />
          </div>

          {/* Под мини-шкалами: слева ИИ рекомендации, справа Похоже на бренды */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="card rounded-2xl p-6">
              <h3 className="text-lg font-semibold mb-3">ИИ рекомендации</h3>
              <ul className="list-disc pl-5 space-y-2 text-sm">
                {recs.map((t,i)=> <li key={i} className="opacity-90">{t}</li>)}
              </ul>
            </div>
            <div className="card rounded-2xl p-6">
              <h3 className="text-lg font-semibold mb-3">Похоже на бренды</h3>
              <ul className="space-y-2">
                {similar.map((b)=> (
                  <li key={b.name} className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                    <span className="text-sm">{b.name}</span>
                    <span className="text-sm font-semibold">{Math.round(b.score)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </header>

        {/* Основные метрики — плавные графики + шкала и число внизу */}
        <MetricPanel
          label="Внимание"
          value={scores.attention}
          series={scores.timeline.attention}
          caption="доля времени с удержанным взглядом"
        />
        <MetricPanel
          label="Вовлечённость"
          value={scores.engagement}
          series={scores.timeline.engagement}
          caption="поведенческий интерес и интеракции"
        />
        <MetricPanel
          label="Запоминаемость"
          value={scores.memory}
          series={scores.timeline.memory}
          caption="вероятность воспоминания через 24ч"
        />
      </main>
    </div>
  );
}

// ---------------- Styles / Theme ----------------
function StyleBrand(){
  return (
    <style>{`
      :root{
        --brand-start:#f6b27a;--brand-mid:#c44fb0;--brand-end:#6b3cff;
        --bg-950:#0b0b12;--bg-900:#111119;--ring:rgba(255,255,255,.08);
        --btn-grad:linear-gradient(90deg,#541BFF,#DB156E);
      }
      .bg-background{
        background:
          radial-gradient(1400px 800px at 20% -10%, rgba(107,60,255,0.35), transparent 65%),
          radial-gradient(1200px 900px at 80% 0%, rgba(246,178,122,0.30), transparent 70%),
          linear-gradient(180deg,var(--bg-950),var(--bg-900));
      }
      .text-foreground{color:#f5f7ff}
      .card{background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01));border:1px solid var(--ring);backdrop-filter:blur(6px)}
      /* animations */
      @keyframes draw{from{stroke-dashoffset:1400}to{stroke-dashoffset:0}}
      @keyframes fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
      .line-anim{stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 1.6s ease-out forwards}
      .fade-in{animation:fadein .6s ease-out both}
    `}</style>
  );
}

// ---------------- Metric Panel (SVG-график с анимацией и ховером) ----------------
function MetricPanel({ label, value, series, caption }){
  const [hover, setHover] = useState(null); // {i,x,y,v}
  const [mounted, setMounted] = useState(false);
  useEffect(()=>{ const t=setTimeout(()=>setMounted(true),10); return ()=>clearTimeout(t); },[]);

  // Размеры и отступы, чтобы подписи не пересекались с графиком
  const W = 1000, H = 300;
  const M = { l: 42, r: 10, t: 10, b: 26 };
  const PW = W - M.l - M.r; // plot width
  const PH = H - M.t - M.b; // plot height
  const yMin = 0, yMax = 100;

  // Шаг по X подбираем так, чтобы подписи были не ближе 40px друг к другу
  const minPxPerTick = 40;
  const roughStep = Math.ceil(((series.length-1) * minPxPerTick) / Math.max(1, PW));
  const stepSec = Math.max(5, Math.ceil(roughStep/5)*5); // кратно 5 сек
  const ticksX = useMemo(()=> Array.from({length: Math.floor((series.length-1)/stepSec)+1}, (_,k)=> k*stepSec), [series.length, stepSec]);
  const yTicks = [0,50,100];

  // Умеренное сглаживание и мягкая кривая
  const smoothed = useMemo(()=> smoothSeries(series, 3), [series]);
  const pathD = useMemo(()=> seriesToSmoothPath(smoothed, yMin, yMax, PW, PH, 0.25), [smoothed, PW, PH]);

  const getXY = (i)=>{
    const x = (i/(series.length-1))*PW;
    const v = Math.max(yMin, Math.min(yMax, series[i] ?? 0));
    const y = (1 - (v - yMin)/(yMax - yMin)) * PH;
    return {x, y, v};
  };

  const onMove = (e)=>{
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const xPlot = Math.max(0, Math.min(PW, relX - M.l));
    const i = Math.max(0, Math.min(series.length-1, Math.round((xPlot / Math.max(1,PW)) * (series.length-1))));
    const {x,y,v} = getXY(i);
    setHover({ i, x: x + M.l, y: y + M.t, v });
  };

  const onLeave = ()=> setHover(null);

  return (
    <div className={`card rounded-2xl p-8 flex flex-col ${mounted ? 'fade-in' : ''}`}>
      <div className="mb-3">
        <h3 className="text-xl font-semibold mb-1">{label}</h3>
        <p className="text-sm opacity-80 mb-3">{caption}</p>
      </div>

      {/* Линейный график на SVG с отступами для подписей осей */}
      <div className="w-full mb-4">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[350px]" onMouseMove={onMove} onMouseLeave={onLeave}>
          {/* Область построения */}
          <g transform={`translate(${M.l},${M.t})`}>
            {/* Вертикальная сетка и подписи времени (сек) */}
            {ticksX.map((t)=> {
              const x = (t/(series.length-1))*PW;
              return (
                <g key={t}>
                  <line x1={x} y1={0} x2={x} y2={PH} stroke="rgba(255,255,255,0.08)" strokeWidth={1}/>
                  {/* подпись X рисуем ниже области графика */}
                  <text x={x} y={PH+16} fill="rgba(255,255,255,0.6)" fontSize="11" textAnchor="middle">{t}s</text>
                </g>
              );
            })}
            {/* Горизонтальная сетка и подписи процентов */}
            {yTicks.map((yt,idx)=> {
              const y = (1 - (yt - yMin)/(yMax - yMin)) * PH;
              return (
                <g key={idx}>
                  <line x1={0} y1={y} x2={PW} y2={y} stroke="rgba(255,255,255,0.08)" strokeWidth={1}/>
                  {/* подпись Y рисуем слева от области графика */}
                  <text x={-8} y={Math.max(12, y-2)} fill="rgba(255,255,255,0.6)" fontSize="11" textAnchor="end">{yt}%</text>
                </g>
              );
            })}
            {/* Линия графика */}
            <path d={pathD} transform="translate(0,0)" fill="none" stroke="#9b8cff" strokeWidth={4} strokeLinejoin="round" strokeLinecap="round" className="line-anim" />
          </g>

          {/* Hover helpers (в абсолютных координатах SVG) */}
          {hover && (
            <>
              <line x1={hover.x} y1={M.t} x2={hover.x} y2={M.t+PH} stroke="rgba(255,255,255,0.25)" strokeWidth={1}/>
              <circle cx={hover.x} cy={hover.y} r={5} fill="#fff" />
              <g>
                <rect x={Math.min(Math.max(hover.x-52,4), W-148)} y={Math.max(hover.y-56, 4)} rx="8" ry="8" width="148" height="50" fill="rgba(21,19,29,0.92)" stroke="rgba(255,255,255,0.15)" />
                <text x={Math.min(Math.max(hover.x-40,12), W-136)} y={Math.max(hover.y-30, 18)} fill="#fff" fontSize="12" fontFamily="Inter, system-ui">
                  <tspan>время: {hover.i}s</tspan>
                </text>
                <text x={Math.min(Math.max(hover.x-40,12), W-136)} y={Math.max(hover.y-12, 36)} fill="#fff" fontSize="12" fontFamily="Inter, system-ui">
                  <tspan>значение: {hover.v}%</tspan>
                </text>
              </g>
            </>
          )}
        </svg>
      </div>

      {/* Шкала прогресса и число внизу */}
      <div className="flex items-center justify-between">
        <div className="w-[90%] h-3 rounded-full bg-white/10 overflow-hidden">
          <div className="h-full" style={{ width:`${value}%`, backgroundImage:'var(--btn-grad)' }} />
        </div>
        <span className="text-sm font-semibold ml-3">{value}%</span>
      </div>
    </div>
  );
}

// ---------------- Mini-card ----------------
function SmallStat({ label, value, caption }){
  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-4">
      <div className="flex items-baseline justify-between">
        <div className="text-sm font-semibold">{label}</div>
        <div className="text-sm font-semibold">{value}%</div>
      </div>
      <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden mt-2">
        <div className="h-full" style={{ width:`${Math.min(100,Math.max(0, value))}%`, backgroundImage:'var(--btn-grad)' }} />
      </div>
      <div className="text-[11px] opacity-70 mt-2">{caption}</div>
    </div>
  );
}

// ---------------- Utils ----------------
function smoothSeries(arr, window=5){
  if(!Array.isArray(arr) || arr.length===0) return [];
  const w = Math.max(1, Math.floor(window));
  if(w===1) return [...arr];
  const half = Math.floor(w/2);
  const out = new Array(arr.length);
  for(let i=0;i<arr.length;i++){
    let sum=0, cnt=0;
    for(let k=-half;k<=half;k++){
      const idx = i+k;
      if(idx>=0 && idx<arr.length){ sum += arr[idx]; cnt++; }
    }
    out[i] = sum / cnt;
  }
  return out;
}

// ---------------- Utils ----------------
function durationToSeconds(dur){
  if(typeof dur !== 'string' || !dur.trim()) return 0;
  const parts = dur.trim().split(':');
  if(parts.length!==1 && parts.length!==2 && parts.length!==3) return 0;
  const nums = parts.map(x=> Number(x));
  if(nums.some(n=> Number.isNaN(n))) return 0;
  if(parts.length===3){ const [h,m,s] = nums; return h*3600 + m*60 + s; }
  if(parts.length===2){ const [m,s] = nums; return m*60 + s; }
  return nums[0];
}

function scoreFromId(id, seconds){
  const seed = id.split('').reduce((a,c)=> a + c.charCodeAt(0), 0);
  const clamp = (x)=> Math.max(0, Math.min(100, Math.round(x)));
  const mkSeries = (bias, freq)=> Array.from({length:seconds}, (_,i)=> clamp(bias + ((seed + i*freq) % 40) - 20));
  const attention = 70, engagement = 65, memory = 68;
  return {
    attention,
    engagement,
    memory,
    timeline:{
      attention: mkSeries(attention,7),
      engagement: mkSeries(engagement,11),
      memory: mkSeries(memory,5)
    }
  };
}

// Преобразование в ломаную (не используется для рендера, оставлено для тестов)
function seriesToPath(series, yMin, yMax){
  if(!Array.isArray(series) || series.length===0) return '';
  const n = series.length;
  const clamp = (x)=> Math.min(Math.max(x, yMin), yMax);
  const mapPoint = (v, i)=> {
    const x = (i/(n-1))*1000;
    const y = (1 - (clamp(v)-yMin)/(yMax-yMin)) * 300;
    return `${i===0?'M':'L'}${x.toFixed(2)},${y.toFixed(2)}`;
  };
  return series.map(mapPoint).join(' ');
}

// Плавная кривая (Catmull-Rom → Bezier)
function seriesToSmoothPath(series, yMin, yMax, W=1000, H=300, tension=0.35){
  if(!Array.isArray(series) || series.length===0) return '';
  const clamp = (x)=> Math.min(Math.max(x, yMin), yMax);
  const pts = series.map((v,i)=>{
    const x = (i/(series.length-1))*W;
    const y = (1 - (clamp(v)-yMin)/(yMax-yMin)) * H;
    return {x,y};
  });
  if(pts.length<2){ return `M0,${H} L${W},${H}`; }
  let d = `M${pts[0].x},${pts[0].y}`;
  for(let i=0;i<pts.length-1;i++){
    const p0 = pts[i-1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i+1];
    const p3 = pts[i+2] || p2;
    const cp1x = p1.x + (p2.x - p0.x) * tension / 6;
    const cp1y = p1.y + (p2.y - p0.y) * tension / 6;
    const cp2x = p2.x - (p3.x - p1.x) * tension / 6;
    const cp2y = p2.y - (p3.y - p1.y) * tension / 6;
    d += ` C${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x},${p2.y}`;
  }
  return d;
}

function similarBrands(name){
  const pool = ['Coca-Cola','Nike','Adidas','Apple','Samsung','BMW','IKEA'];
  const b = name.length;
  return pool.slice(0,5).map((p,i)=> ({ name:p, score: 80 - i*10 + (b%7) }));
}

function generateAIRecommendations(){
  return [
    'Первые секунды не цепляют: рекомендуется добавить хук.',
    'Средняя вовлечённость, улучшите динамику середины.',
    'Запоминаемость стабильная, но финал слаб — добавьте бренд-кадр.'
  ];
}

// ---------------- Lightweight dev tests ----------------
function runDevTests(){
  try{
    console.assert(durationToSeconds('00:15')===15, '00:15 -> 15s');
    console.assert(durationToSeconds('01:00')===60, '01:00 -> 60s');
    console.assert(durationToSeconds('1:02:03')===3723, '1:02:03 -> 3723s');
    const p = seriesToPath([0,50,100],0,100);
    console.assert(p.startsWith('M0.00,'), 'path starts with M');
  }catch(e){ /* no-op */ }
}
