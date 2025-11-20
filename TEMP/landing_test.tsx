import React, { useState, useEffect, useRef, useMemo } from "react";
import { LogIn, UploadCloud, BarChart3, Brain, LineChart, Shield, Gauge, Sparkles, X, CreditCard, CalendarClock, LogOut, ChevronRight, CheckCircle2, AlertCircle, Clock, Download, FileText } from "lucide-react";

// =====================================
// Root app with simple in-memory routing
// =====================================
export default function AppRouter(){
  const [view, setView] = useState<'landing'|'dashboard'>('landing');
  const handleAuthedToDashboard = () => setView('dashboard');
  return view==='dashboard' ? <AccountDashboardInline onLogout={()=>setView('landing')}/> : <SynapSightLandingFull onAuthed={handleAuthedToDashboard}/>;
}

// ===============================
// Landing (unchanged look & feel)
// ===============================
function SynapSightLandingFull({ onAuthed }:{ onAuthed:()=>void }){
  const [showReg, setShowReg] = useState(false);
  // New: checkout flow after "Попробовать"
  const [pendingAction, setPendingAction] = useState<'none'|'checkout'>('none');
  const [chosenPlan, setChosenPlan] = useState<string|undefined>(undefined);
  const [showChangePlan, setShowChangePlan] = useState(false);
  const [showAddCard, setShowAddCard] = useState(false);

  // After auth, branch by pending action
  const handleAuthSubmit = () => {
    if (pendingAction === 'checkout') {
      setShowReg(false);
      setShowChangePlan(true);
    } else {
      setShowReg(false);
      onAuthed();
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-clip">
      <StyleBrand />
      <Header onLogin={()=>{ setPendingAction('none'); setShowReg(true); }} />
      <main>
        <AboveTheFold onRequireAuth={()=>{ setPendingAction('none'); setShowReg(true); }} />
        <Features />
        <Metrics />
        <Pricing onTry={(plan)=>{ setChosenPlan(plan); setPendingAction('checkout'); setShowReg(true); }} />
        <CTA />
        <FAQ />
      </main>
      <Footer />

      {/* Auth first, then plan & card if came from "Попробовать" */}
      <AuthModal
        open={showReg}
        onClose={()=>setShowReg(false)}
        onSubmit={handleAuthSubmit}
      />
      <ChangePlanModal
        open={showChangePlan}
        onClose={()=>setShowChangePlan(false)}
        onPick={(plan)=>{ setChosenPlan(plan); setShowChangePlan(false); setShowAddCard(true); }}
        preset={chosenPlan}
      />
      <AddCardModal
        open={showAddCard}
        onClose={()=>{ setShowAddCard(false); onAuthed(); }}
      />
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
          radial-gradient(1000px 900px at 50% 100%, rgba(219,21,110,0.25), transparent 75%),
          radial-gradient(1200px 700px at 10% 70%, rgba(84,27,255,0.28), transparent 75%),
          radial-gradient(1100px 800px at 90% 80%, rgba(255,90,210,0.22), transparent 75%),
          radial-gradient(1000px 600px at 50% 50%, rgba(200,120,255,0.25), transparent 70%),
          radial-gradient(1000px 700px at 40% 20%, rgba(255,220,170,0.18), transparent 70%),
          linear-gradient(180deg,var(--bg-950),var(--bg-900));
      }
      .above-gradient{
        background:
          radial-gradient(800px 500px at 50% 0%, rgba(107,60,255,0.35), transparent 70%),
          radial-gradient(900px 700px at 10% 80%, rgba(219,21,110,0.25), transparent 70%),
          radial-gradient(1000px 800px at 90% 50%, rgba(255,200,140,0.18), transparent 75%),
          linear-gradient(180deg,var(--bg-950),var(--bg-900));
      }
      .text-foreground{color:#f5f7ff}
      .brand-text{background:linear-gradient(90deg,var(--brand-start),var(--brand-mid),var(--brand-end));-webkit-background-clip:text;background-clip:text;color:transparent}
      .card{background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01));border:1px solid var(--ring);backdrop-filter:blur(6px)}
      .btn-main{background-image:var(--btn-grad);color:#fff}
      html{scroll-behavior:smooth}
      .scroll-hint{display:flex;flex-direction:column;align-items:center;gap:6px;margin:12px 0 24px;transition:opacity 0.5s ease;}
      .scroll-dot{width:10px;height:10px;border-radius:9999px;background:#fff;opacity:.75;animation:sd 1.6s ease-in-out infinite}
      @keyframes sd{0%{transform:translateY(0);opacity:.4}50%{transform:translateY(8px);opacity:1}100%{transform:translateY(0);opacity:.4}}
    `}</style>
  );
}

// ---------------- Header ----------------
function Header({ onLogin }:{ onLogin:()=>void }) {
  return (
    <header className="sticky top-0 z-30 backdrop-blur bg-[rgba(21,19,29,0.8)] h-[64px] flex.items-center">
      <div className="mx-auto max-w-7xl px-6 flex flex-row items-center justify-between w-full h-full">
        <div className="flex flex-row items-center gap-2 whitespace-nowrap">
          <span className="font-bold text-2xl tracking-tight">Synap</span>
          <span className="brand-text font-semibold text-2xl tracking-tight">Sight</span>
          <span className="text-[11px] ml-3 hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/10 text-white border border-white/15">нейромаркетинг</span>
        </div>
        <nav className="hidden lg:flex flex-row items-center gap-8 text-sm text-gray-200/85">
          <a href="#upload" className="hover:text-white transition-colors">Продукт</a>
          <a href="#features" className="hover:text-white transition-colors">Как это работает</a>
          <a href="#metrics" className="hover:text-white transition-colors">Метрики</a>
          <a href="#pricing" className="hover:text-white transition-colors">Тарифы</a>
          <a href="#faq" className="hover:text-white transition-colors">FAQ</a>
        </nav>
        <div className="flex flex-row items-center gap-3">
          <a href="#upload" className="inline-flex items-center gap-2 rounded-2xl px-5 py-2 font-semibold text-white shadow-md btn-main">Проверить рекламу</a>
          <button onClick={onLogin} className="inline-flex items-center gap-2 rounded-2xl px-4 py-1.5 border border-white/15 hover:border-white/30 text-sm transition-colors"><LogIn className="w-4 h-4" /> Войти</button>
        </div>
      </div>
    </header>
  );
}

// ---------------- Above the fold ----------------
function AboveTheFold({ onRequireAuth }:{ onRequireAuth:()=>void }){
  return (
    <section id="above" className="min-h-[calc(100vh-64px)] flex flex-col justify-center relative overflow-hidden above-gradient">
      <Hero />
      <QuickUpload onRequireAuth={onRequireAuth} />
      <ScrollHint />
    </section>
  );
}

function Hero() {
  return (
    <section className="text-center pt-16 pb-10 px-5">
      <div className="inline-flex items-center gap-2 text-xs text-gray-300/90 bg-white/5 border border-white/10 rounded-full px-3 py-1">AI-претест видеорекламы</div>
      <h1 className="mt-5 text-5xl md:text-6xl font-black leading-tight">Проверьте креатив <span style={{backgroundImage:'var(--btn-grad)',WebkitBackgroundClip:'text',backgroundClip:'text',color:'transparent'}}>до запуска</span></h1>
      <p className="mt-4 text-lg md:text-xl text-gray-200/85 max-w-3xl mx-auto">SynapSight анализирует видеоролик и прогнозирует <span className="font-semibold">нейроэффективность</span> на основе моделей внимания, вовлеченности и запоминаемости.</p>
    </section>
  );
}

function ScrollHint(){
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const handleScroll = () => { setVisible(window.scrollY <= 100); };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  return (
    <div className={`scroll-hint text-center ${visible ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
      <span className="text-[11px] opacity-70">Прокрутите вниз</span>
      <span className="scroll-dot" />
    </div>
  );
}

// ---------------- Quick Upload (hero area) ----------------
function QuickUpload({ onRequireAuth }:{ onRequireAuth:()=>void }){
  const inputRef = useRef<HTMLInputElement|null>(null);
  const [drag, setDrag] = useState(false);
  const [uploads, setUploads] = useState<{id:string; name:string; progress:number; status:'uploading'|'ready'}[]>([]);
  const timersRef = useRef<Record<string, any>>({});
  const hasPromptedRef = useRef(false);

  useEffect(()=>()=>{ Object.values(timersRef.current).forEach((t)=> clearInterval(t)); },[]);

  useEffect(()=>{
    if (!hasPromptedRef.current && uploads.some(u=>u.status==='ready')){
      hasPromptedRef.current = true; onRequireAuth?.();
    }
  }, [uploads, onRequireAuth]);

  const onPick = () => inputRef.current?.click();
  const handleFiles = (fileList: FileList | null) => { Array.from(fileList||[]).forEach((f)=> startUpload(f)); };
  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => { handleFiles(e.target.files); e.target.value = ""; };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files); };

  const startUpload = (file: File) => {
    const id = `${Date.now()}_${Math.random().toString(36).slice(2,7)}`;
    const name = file?.name || 'video.mp4';
    setUploads((prev) => [...prev, { id, name, progress: 0, status: 'uploading' }]);

    const step = Math.floor(6 + Math.random()*10);
    let reached = false;
    const timer = setInterval(() => {
      setUploads((prev) => prev.map(u => {
        if (u.id !== id) return u;
        const next = Math.min(100, u.progress + step);
        if (next >= 100) reached = true;
        return { ...u, progress: next, status: next >= 100 ? 'ready' : 'uploading' };
      }));
      if (reached) { clearInterval(timer); delete timersRef.current[id]; }
    }, 180);
    timersRef.current[id] = timer;
  };

  return (
    <section id="upload" className="mx-auto max-w-6xl px-5.pb-6">
      <div className="card rounded-3xl p-10">
        <div className="text-sm text-gray-300/85 mb-4">Быстрый старт</div>
        <div
          onDragOver={(e)=>{e.preventDefault(); setDrag(true);}}
          onDragLeave={()=>setDrag(false)}
          onDrop={onDrop}
          onClick={onPick}
          className={`text-center border-2 border-dashed rounded-xl p-12 transition ${drag? 'border-white/60 bg-white/10':'border-white/20 hover:border-white/40 hover:bg-white/5'}`}
          role="button" aria-label="Выбрать файл для загрузки"
        >
          <UploadCloud className="w-10 h-10 mx-auto mb-2" />
          <p className="text-sm text-gray-300/85">Перетащите видео или выберите файл для анализа (.mp4 / .mov)</p>
          <button type="button" className="mt-4 inline-flex items-center gap-2 px-4 py-2 border border-white/20 rounded-xl hover:border-white/40">
            <UploadCloud className="w-4 h-4" /> Выбрать файл
          </button>
          <input ref={inputRef} type="file" className="hidden" accept="video/mp4,video/quicktime" multiple onChange={onChange} />
        </div>

        {uploads.length > 0 && (
          <div className="mt-8 space-y-4">
            {uploads.map((u) => (
              <div key={u.id} className="relative rounded-xl border border-white/12 bg-white/5 p-5 flex items-center gap-4">
                <div className="shrink-0 w-14 h-10 rounded bg-white/10 flex items-center justify-center text-[10px] opacity-80">MP4</div>
                <div className="min-w-0 grow">
                  <div className="text-base font-medium truncate">{u.name}</div>
                  {u.status === 'uploading' ? (
                    <div className="mt-2">
                      <ProgressBar value={u.progress} />
                      <div className="text-[12px] opacity-60 mt-1">Загрузка… {u.progress}%</div>
                    </div>
                  ) : (
                    <div className="text-[12px] opacity-60 mt-1">Загружено • готово к анализу</div>
                  )}
                </div>
                {u.status === 'ready' && (
                  <button className="ml-auto inline-flex items-center gap-1 text-[13px] px-4 py-2 rounded-lg border border-white/12 bg-white/5 hover:bg-white/10 transition-colors">
                    <BarChart3 className="w-4 h-4"/> Смотреть метрики
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-10 text-center text-sm text-gray-200/85">
          <div className="card rounded-xl py-3">Прогноз нейроэффективности</div>
          <div className="card rounded-xl py-3">Карты внимания и ритм монтажа</div>
          <div className="card rounded-xl py-3">Детекция бренда и объектов</div>
          <div className="card rounded-xl py-3">Рекомендации по улучшению</div>
        </div>
      </div>
    </section>
  );
}

function ProgressBar({ value }:{ value:number }){
  return (
    <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(100, value||0))}%`, backgroundImage: 'var(--btn-grad)' }} />
    </div>
  );
}

// ---------------- Features ----------------
function Features() {
  return (
    <section id="features" className="mx-auto max-w-6xl px-5 py-16 space-y-10">
      <h2 className="text-3xl font-bold text-center mb-10">Профессиональный анализ контента за считанные минуты</h2>
      <div className="grid md:grid-cols-2 gap-8 items-center">
        <div className="aspect-[16/10] bg-white/10 rounded-2xl" />
        <div>
          <h3 className="text-2xl font-semibold mb-2 flex items-center gap-2"><Brain className="w-6 h-6" /> Нейро‑прогноз</h3>
          <p className="text-gray-300">Оцениваем вероятность запоминания, уровень внимания и вовлечённости по видеосигнатурам, обученным на физиологических данных.</p>
        </div>
      </div>
      <div className="grid md:grid-cols-2 gap-8 items-center">
        <div>
          <h3 className="text-2xl font-semibold mb-2 flex items-center gap-2"><LineChart className="w-6 h-6" /> Профиль креатива</h3>
          <p className="text-gray-300">Монтажный ритм, плотность событий, аудио‑энергетика, объектная сцена и эмоциональный тон — собираем в единый профиль.</p>
        </div>
        <div className="aspect-[16/10] bg-white/10 rounded-2xl" />
      </div>
      <div className="grid md:grid-cols-2 gap-8 items-center">
        <div className="aspect-[16/10] bg-white/10 rounded-2xl" />
        <div>
          <h3 className="text-2xl font-semibold mb-2 flex items-center gap-2"><Shield className="w-6 h-6" /> Рекомендации</h3>
          <p className="text-gray-300">Подсвечиваем слабые места креатива и предлагаем правки: усилить бренд‑кадр, изменить темп, укоротить сцену и пр.</p>
        </div>
      </div>
    </section>
  );
}

// ---------------- Metrics ----------------
function Metrics() {
  return (
    <section id="metrics" className="mx-auto max-w-6xl px-5 py-16">
      <h2 className="text-3xl font-bold text-center mb-8">Ключевые метрики</h2>
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Metric icon={Gauge} label="Внимание" value="Высокое" />
        <Metric icon={BarChart3} label="Вовлечённость" value="Средняя" />
        <Metric icon={Sparkles} label="Запоминаемость" value="Выше среднего" />
        <Metric icon={Shield} label="Brand-safety" value="Ок" />
      </div>
    </section>
  );
}

function Metric({ icon: Icon, label, value }:{ icon:any; label:string; value:string }) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-white/10 bg-black/20 px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center bg-white/10"><Icon className="w-4 h-4" /></div>
        <div className="text-sm opacity-80">{label}</div>
      </div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}

// ---------------- Pricing ----------------
function Pricing({ onTry }:{ onTry:(plan:string)=>void }) {
  return (
    <section id="pricing" className="mx-auto max-w-6xl px-5 py-16 text-center">
      <h2 className="text-3xl font-bold mb-8">Тарифы (кредиты по хронометражу)</h2>
      <div className="grid md:grid-cols-3 gap-8">
        <PriceCard title="S" note="короткие ролики" price="5 000 ₽ / мес" features={["до 30 сек", "быстрый отчёт", "доступ к базовым метрикам"]} onTry={()=>onTry('S')} />
        <PriceCard title="M" note="универсальный" price="10 000 ₽ / мес" features={["30–60 сек", "все метрики", "рекомендации по правкам"]} featured onTry={()=>onTry('M')} />
        <PriceCard title="L" note="длинные ролики" price="20 000 ₽ / мес" features={["60–120 сек", "приоритетная обработка", "API-доступ (опц.)"]} onTry={()=>onTry('L')} />
      </div>
    </section>
  );
}

function PriceCard({ title, note, price, features = [], featured, onTry }:{ title:string; note:string; price:string; features?:string[]; featured?:boolean; onTry?:()=>void }) {
  const items = Array.isArray(features) ? features : [];
  const borderWrap = featured ? 'border-2 border-transparent bg-gradient-to-r from-[#541BFF] to-[#DB156E] p-[1px]' : 'border border-white/10';
  return (
    <div className={`rounded-2xl ${borderWrap}`}>
      <div className="card rounded-2xl p-6">
        <div className="text-sm text-gray-400 mb-1">{note}</div>
        <div className="text-4xl font-extrabold mb-2">{title}</div>
        <div className="text-lg font-semibold mb-3">{price}</div>
        <ul className="text-sm text-gray-300 space-y-1 mb-6">
          {items.map((f, i) => <li key={i}>• {f}</li>)}
        </ul>
        <button onClick={onTry} className="border border-white/70 text.white rounded-xl px-4 py-2 hover:bg-white/10 transition">Попробовать</button>
      </div>
    </div>
  );
}

// ---------------- CTA ----------------
function CTA() {
  return (
    <section className="mx-auto max-w-6xl px-5 py-20 text-center">
      <div className="card rounded-3xl p-10">
        <h3 className="text-3xl font-bold mb-4">Готовы проверить креатив?</h3>
        <p className="text-sm text-gray-300 mb-6">Загрузите ролик — и получите прогноз нейроэффективности за пару минут.</p>
        <div className="flex flex-wrap justify-center gap-4">
          <a href="#pricing" className="px-6 py-3 rounded-2xl text-white font-semibold shadow-md btn-main">Проверить рекламу сейчас</a>
          <a href="#features" className="px-6 py-3 rounded-2xl border border-white/10 hover:border-white/20 transition">Подробнее о метриках</a>
        </div>
      </div>
    </section>
  );
}

// ---------------- FAQ (multi-open) ----------------
function FAQ() {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const faqs = [
    { q: 'Как работает анализ видео?', a: 'Система анализирует видеосигнатуры и рассчитывает показатели внимания, вовлеченности и запоминаемости.' },
    { q: 'Какие форматы поддерживаются?', a: 'Поддерживаются .mp4 и .mov, а также ссылки на видео.' },
    { q: 'Можно ли загрузить несколько роликов?', a: 'Да, мультизагрузка поддерживается — каждый ролик обрабатывается отдельно.' },
  ];
  const toggle = (i:number)=> setOpen(prev=>{ const next=new Set(prev); next.has(i)?next.delete(i):next.add(i); return next; });
  return (
    <section id="faq" className="mx-auto max-w-6xl px-5 py-16">
      <h2 className="text-3xl font-bold text-center mb-8">Часто задаваемые вопросы</h2>
      <div className="space-y-4">
        {faqs.map((item, i) => {
          const isOpen = open.has(i);
          return (
            <div key={i} className="card rounded-xl p-6 cursor-pointer transition-all hover:border-white/20" onClick={()=>toggle(i)}>
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">{item.q}</h4>
                <span className="text-xl select-none" style={{ transform:`rotate(${isOpen?90:0}deg)` }}>›</span>
              </div>
              {isOpen && <p className="text-sm text-gray-300 mt-2">{item.a}</p>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------- Auth Modal ----------------
function AuthModal({ open, onClose, onSubmit }:{ open:boolean; onClose:()=>void; onSubmit:()=>void }){
  const [mode, setMode] = React.useState<'login'|'register'|'success'>('login');
  const [step, setStep] = React.useState<1|2|3>(1);
  const [loading, setLoading] = React.useState(false);
  const [form, setForm] = React.useState({ email:'', password:'', company:'', agree:false, code:'' });

  React.useEffect(()=>{
    if (!open) { setMode('login'); setStep(1); setLoading(false); setForm({ email:'', password:'', company:'', agree:false, code:'' }); }
  }, [open]);

  if (!open) return null;

  const submitLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(()=>{ setLoading(false); onSubmit(); }, 700);
  };

  const nextFromStep1 = (e: React.FormEvent) => { e.preventDefault(); setStep(2); };
  const nextFromStep2 = (e: React.FormEvent) => { e.preventDefault(); if (!form.agree) return; setStep(3); };
  const completeRegistration = (e: React.FormEvent) => {
    e.preventDefault();
    setMode('success');
    setTimeout(()=>{ onSubmit(); }, 1100);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.92)] border-white/20">
        {/* Close */}
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4"/></button>

        {mode === 'success' ? (
          <div className="flex flex-col items-center justify-center py-10">
            <style>{`@keyframes pop{0%{transform:scale(.8);opacity:.2}60%{transform:scale(1.05);opacity:1}100%{transform:scale(1)}}`}</style>
            <CheckCircle2 className="w-16 h-16 text-emerald-400" style={{ animation:'pop .9s.ease-out' }} />
            <div className="mt-3 text-lg font-semibold">Регистрация подтверждена</div>
            <div className="text-sm opacity-80">Продолжаем…</div>
          </div>
        ) : mode === 'login' ? (
          <>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xl font-bold">Вход</h3>
            </div>
            <form onSubmit={submitLogin} className="space-y-3">
              <input required placeholder="Email" type="email" value={form.email} onChange={e=>setForm(f=>({...f,email:e.target.value}))} className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
              <input required placeholder="Пароль" type="password" value={form.password} onChange={e=>setForm(f=>({...f,password:e.target.value}))} className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
              <button disabled={loading} className="w-full rounded-xl px-4 py-2 font-semibold btn-main">{loading? 'Входим…' : 'Войти'}</button>
            </form>
            <button className="w-full text-xs opacity-80 mt-3 underline underline-offset-4" onClick={()=>{ setMode('register'); setStep(1); }}>Зарегистрироваться</button>
          </>
        ) : (
          <>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xl font-bold">Регистрация</h3>
              <div className="text-xs opacity-70">Шаг {step} из 3</div>
            </div>

            {step === 1 && (
              <form className="space-y-3" onSubmit={nextFromStep1}>
                <input required placeholder="Email" type="email" value={form.email} onChange={e=>setForm(f=>({...f,email:e.target.value}))} className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
                <input required placeholder="Пароль" type="password" value={form.password} onChange={e=>setForm(f=>({...f,password:e.target.value}))} className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
                <button className="w-full rounded-xl px-4 py-2 font-semibold.btn-main">Далее</button>
              </form>
            )}

            {step === 2 && (
              <form className="space-y-3" onSubmit={nextFromStep2}>
                <input placeholder="Компания (необязательно)" value={form.company} onChange={e=>setForm(f=>({...f,company:e.target.value}))} className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
                <label className="flex items-center gap-2 text-sm opacity-90">
                  <input type="checkbox" checked={form.agree} onChange={e=>setForm(f=>({...f,agree:e.target.checked}))} />
                  Соглашаюсь с условиями и политикой конфиденциальности
                </label>
                <div className="flex gap-2">
                  <button type="button" onClick={()=>setStep(1)} className="w-1/3 rounded-xl px-4 py-2 border border-white/15 hover:border-white/30">Назад</button>
                  <button disabled={!form.agree} className="w-2/3 rounded-xl px-4 py-2 font-semibold btn-main">Далее</button>
                </div>
              </form>
            )}

            {step === 3 && (
              <form className="space-y-3" onSubmit={completeRegistration}>
                <p className="text-sm opacity-85">Мы отправили код подтверждения на <span className="font-medium">{form.email || 'ваш email'}</span>.</p>
                <input required placeholder="Код из письма" value={form.code} onChange={e=>setForm(f=>({...f,code:e.target.value}))} className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
                <div className="flex gap-2">
                  <button type="button" onClick={()=>setStep(2)} className="w-1/3 rounded-xl px-4 py-2 border border-white/15 hover:border-white/30">Назад</button>
                  <button className="w-2/3 rounded-xl px-4 py-2 font-semibold btn-main">Завершить</button>
                </div>
              </form>
            )}

            <button className="w-full text-xs opacity-80 mt-3 underline underline-offset-4" onClick={()=>{ setMode('login'); }}>У меня уже есть аккаунт</button>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------- Footer ----------------
function Footer() {
  return (
    <footer className="mx-auto max-w-7xl px-5 py-12 text-gray-300 text-sm">
      <div className="grid md:grid-cols-3 gap-8 border-t border-white/10 pt-8">
        <div>
          <div className="flex items-baseline gap-2 mb-3">
            <span className="font-bold text-xl">Synap</span>
            <span className="brand-text font-semibold text-xl">Sight</span>
          </div>
          <p className="text-gray-400 text-sm mb-3 max-w-sm">Платформа претестирования видеорекламы с AI‑анализом креатива и прогнозом нейроэффективности.</p>
          <p className="text-xs opacity-80">© {new Date().getFullYear()} SynapSight. Все права защищены.</p>
        </div>
        <div>
          <h5 className="font-semibold mb-2">Меню</h5>
          <ul className="space-y-1">
            <li><a href="#upload" className="hover:text-white">Продукт</a></li>
            <li><a href="#features" className="hover:text-white">Как это работает</a></li>
            <li><a href="#metrics" className="hover:text-white">Метрики</a></li>
            <li><a href="#pricing" className="hover:text-white">Тарифы</a></li>
            <li><a href="#faq" className="hover:text-white">FAQ</a></li>
          </ul>
        </div>
        <div>
          <h5 className="font-semibold mb-2">Контакты</h5>
          <p className="text-xs mb-1">Email: <a href="mailto:hello@synapsight.io" className="underline">hello@synapsight.io</a></p>
          <p className="text-xs mb-1">Telegram: @synapsight</p>
          <p className="text-xs opacity-70">ООО «СинапСайт» · ОГРН ХХХХХХХХХХХХ · ИНН ХХХХХХХХХ</p>
        </div>
      </div>
    </footer>
  );
}

// ======================================================
// Inline Account Dashboard (copied & trimmed to be local)
// ======================================================
function AccountDashboardInline({ onLogout }:{ onLogout:()=>void }){
  const [user] = useState({
    name: "Анна Маркетолог",
    plan: { title: "Универсальный", price: "10 000 ₽ / мес" },
    billing: { nextChargeISO: addDaysISO(11) },
    usage: { usedMin: 18, quotaMin: 60 },
  });
  const [uploads, setUploads] = useState<UploadCardData[]>(sampleUploads());
  const [showChangePlan, setShowChangePlan] = useState(false);
  const [showAddCard, setShowAddCard] = useState(false);
  const [showInvoices, setShowInvoices] = useState(false);
  const usagePct = Math.min(100, Math.round(user.usage.usedMin / user.usage.quotaMin * 100));
  const invoices = useMemo(()=> sampleInvoices(), []);

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-clip">
      <StyleBrand/>
      <AccountHeader userName={user.name} onLogout={onLogout} />
      <main className="mx-auto max-w-7xl px-5 py-8 space-y-8">
        <section className="grid lg:grid-cols-1 gap-6">
          <div className="card rounded-2xl p-5 lg:col-span-1">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="min-w-[260px]">
                <h1 className="text-2xl font-bold">Личный кабинет</h1>
                <p className="text-sm text-gray-300/85">Здравствуйте, {user.name.split(' ')[0]}! Здесь ваш план, использование и загрузки.</p>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={()=>setShowAddCard(true)} className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/15 hover:border-white/30 text-sm"><CreditCard className="w-4 h-4"/> Карта</button>
                <button onClick={()=>setShowInvoices(true)} className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/15 hover:border-white/30 text-sm"><FileText className="w-4 h-4"/> Платёжные документы</button>
              </div>
            </div>
            <div className="mt-6 grid md:grid-cols-3 gap-4">
              <div className="relative p-4 rounded-xl border border-white/10 bg-black/20">
                <div className="absolute top-3 right-3">
                  <button onClick={()=>setShowChangePlan(true)} className="text-xs px-2 py-1 rounded-lg border border-white/15 hover:border-white/30">Изменить тариф</button>
                </div>
                <div className="text-xs opacity-80">План</div>
                <div className="text-lg font-semibold mt-1">{user.plan.title}</div>
                <div className="text-xs opacity-70 mt-0.5">{user.plan.price}</div>
                <div className="mt-2 text-xs opacity-80 inline-flex items-center gap-1"><CalendarClock className="w-3.5 h-3.5"/> Следующее списание: {formatDate(user.billing.nextChargeISO)}</div>
              </div>
              <div className="p-4 rounded-xl border border-white/10 bg-black/20 md:col-span-2">
                <div className="text-xs opacity-80 mb-2">Использование минут</div>
                <UsageBar pct={usagePct} />
                <div className="mt-2 text-xs opacity-70">{user.usage.usedMin} из {user.usage.quotaMin} мин</div>
              </div>
            </div>
          </div>
        </section>
        <section>
          <div className="card rounded-3xl p-8 space-y-6">
            <UploadAreaDash onFilesReady={(items)=> setUploads((prev)=>[...items,...prev])} />
            <div className="space-y-3">
              {uploads.map((u)=> (
                <UploadCard key={u.id} data={u} onOpenMetrics={()=> alert(`Открыть метрики для ${u.name}`)} />
              ))}
            </div>
          </div>
        </section>
      </main>
      <ChangePlanModal open={showChangePlan} onClose={()=>setShowChangePlan(false)} />
      <AddCardModal open={showAddCard} onClose={()=>setShowAddCard(false)} />
      <InvoicesModal open={showInvoices} onClose={()=>setShowInvoices(false)} invoices={invoices} />
      <Footer />
    </div>
  );
}

// ---- Types for dashboard ----
type UploadCardData = { id:string; name:string; duration:string; createdISO:string; status:'ready'|'processing'|'queued'|'error' };

// ---- Dashboard header ----
function AccountHeader({ userName, onLogout }:{ userName:string; onLogout:()=>void }){
  return (
    <header className="sticky top-0 z-30 backdrop-blur bg-[rgba(21,19,29,0.8)] h-[64px] flex items-center">
      <div className="mx-auto max-w-7xl px-5 w-full flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-bold text-xl">Synap</span><span className="brand-text font-semibold text-xl">Sight</span>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="hidden sm:inline opacity-80">{userName}</span>
          <button onClick={onLogout} className="inline-flex items.center gap-2 px-3 py-1.5 rounded-xl border border-white/15 hover:border-white/30"><LogOut className="w-4 h-4"/> Выйти</button>
        </div>
      </div>
    </header>
  );
}

// ---- Upload area in dashboard ----
function UploadAreaDash({ onFilesReady }:{ onFilesReady:(items:UploadCardData[])=>void }){
  const inputRef = useRef<HTMLInputElement|null>(null);
  const [drag, setDrag] = useState(false);
  const [queue, setQueue] = useState<{id:string; name:string; progress:number; done:boolean;}[]>([]);
  const timersRef = useRef<Record<string, any>>({});
  useEffect(()=>()=>{ Object.values(timersRef.current).forEach(clearInterval); },[]);
  const onPick = () => inputRef.current?.click();

  const handleFiles = (fileList: FileList | null) => {
    const files = Array.from(fileList ?? []);
    files.forEach((file)=>{
      const id = `${Date.now()}_${Math.random().toString(36).slice(2,7)}`;
      setQueue(prev=> [...prev, { id, name:file.name, progress:0, done:false }]);
      const speed = Math.floor(6 + Math.random()*10);
      let t:any;
      const finish = () => {
        const ready: UploadCardData = { id, name:file.name, duration:'—:—', createdISO:new Date().toISOString(), status:'queued' };
        onFilesReady([ready]);
        setQueue(p=> p.filter(q=> q.id!==id));
        clearInterval(t); delete timersRef.current[id];
      };
      t = setInterval(()=>{
        setQueue(prev=>{
          const updated = prev.map(item=>{
            if(item.id!==id) return item; const next=Math.min(100,item.progress+speed); return { ...item, progress:next, done: next>=100 };
          });
          const was = prev.find(i=>i.id===id)?.done ?? false;
          const now = updated.find(i=>i.id===id)?.done ?? false;
          if(!was && now){ setTimeout(finish,0); }
          return updated;
        });
      }, 180);
      timersRef.current[id]=t;
    });
  };

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => { handleFiles(e.target.files); e.target.value = ""; };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files); };

  return (
    <div
      onDragOver={(e)=>{e.preventDefault(); setDrag(true);}}
      onDragLeave={()=>setDrag(false)}
      onDrop={onDrop}
      onClick={onPick}
      className={`text-center border-2 border-dashed rounded-xl p-10.transition ${drag? 'border-white/60 bg-white/10':'border-white/20 hover:border-white/40 hover:bg-white/5'}`}
      role="button" aria-label="Выбрать файл"
    >
      <UploadCloud className="w-10 h-10 mx-auto mb-2" />
      <p className="text-sm text-gray-300/85">Перетащите видео или выберите файл для анализа (.mp4 / .mov)</p>
      <button type="button" className="mt-4 inline-flex items-center gap-2 px-4 py-2 border border-white/20 rounded-xl hover:border-white/40">
        <UploadCloud className="w-4 h-4" /> Выбрать файл
      </button>
      <input ref={inputRef} type="file" className="hidden" accept="video/mp4,video/quicktime" multiple onChange={onChange} />

      {queue.length>0 && (
        <div className="mt-4 space-y-3">
          {queue.map((q)=> (
            <div key={q.id} className="rounded-xl border border-white/12 bg-white/5 p-4">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium truncate mr-3">{q.name}</div>
                <div className="text-xs opacity-70">{q.progress}%</div>
              </div>
              <div className="mt-2 w-full h-2 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full" style={{ width: `${q.progress}%`, backgroundImage: 'var(--btn-grad)' }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Small UI reused ----
function UsageBar({ pct }:{ pct:number }){ return (<div className="w-full h-2 rounded-full bg-white/10 overflow-hidden"><div className="h-full" style={{ width: `${pct}%`, backgroundImage: 'var(--btn-grad)' }} /></div>); }
function UploadCard({ data, onOpenMetrics }:{ data:UploadCardData; onOpenMetrics: ()=>void }){
  return (
    <div className="rounded-xl border border-white/12 bg-white/5 p-5 flex items-center gap-4">
      <div className="shrink-0 w-14 h-10 rounded bg-white/10 flex items-center justify-center text-[10px] opacity-80">MP4</div>
      <div className="min-w-0 grow">
        <div className="text-base font-medium truncate">{data.name}</div>
        <div className="text-[12px] opacity-60 truncate">{formatDateTime(data.createdISO)} • {data.duration}</div>
      </div>
      <StatusPill status={data.status} />
      <button className="ml-auto inline-flex items-center gap-1 text-[13px] px-4 py-2 rounded-lg border border-white/12 bg.white/5 hover:bg-white/10 transition-colors" onClick={onOpenMetrics}>
        <BarChart3 className="w-4 h-4"/> Смотреть метрики
      </button>
      <button className="p-2 rounded-lg border border-white/10 hover:border-white/25"><ChevronRight className="w-4 h-4"/></button>
    </div>
  );
}
function StatusPill({ status }:{ status:UploadCardData['status'] }){
  if(status==='ready') return <span className="inline-flex items-center gap-1 text-emerald-300/90 text-xs"><CheckCircle2 className="w-4 h-4"/> Готово</span>;
  if(status==='processing') return <span className="inline-flex items-center gap-1 text-amber-300/90 text-xs"><Clock className="w-4 h-4"/> Обработка</span>;
  if(status==='queued') return <span className="inline-flex items-center gap-1 text-sky-300/90 text-xs"><Clock className="w-4 h-4"/> В очереди</span>;
  return <span className="inline-flex items-center gap-1 text-rose-300/90 text-xs"><AlertCircle className="w-4 h-4"/> Ошибка</span>;
}

// ---- Modals for dashboard ----
function ModalBase({ children, onClose }:{ children:React.ReactNode; onClose:()=>void }){
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-2xl card rounded-2xl p-6 bg-[rgba(21,19,29,0.92)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4"/></button>
        {children}
      </div>
    </div>
  );
}

function ChangePlanModal({ open, onClose, onPick, preset }:{ open:boolean; onClose:()=>void; onPick?:(plan:string)=>void; preset?:string }){
  if(!open) return null;
  const plans = [
    {title:'S', note:'короткие ролики', price:'5 000 ₽ / мес'},
    {title:'M', note:'универсальный', price:'10 000 ₽ / мес'},
    {title:'L', note:'длинные ролики', price:'20 000 ₽ / мес'}
  ];
  return (
    <ModalBase onClose={onClose}>
      <h3 className="text-xl font-bold mb-3">Выберите тариф</h3>
      <div className="grid sm:grid-cols-3 gap-4">
        {plans.map((p)=> (
          <div key={p.title} className={`card rounded-2xl p-4 ${preset===p.title? 'ring-1 ring-white/40': ''}`}>
            <div className="text-sm text-gray-400">{p.note}</div>
            <div className="text-3xl font-extrabold mt-1">{p.title}</div>
            <div className="text-sm opacity-80 mt-2">{p.price}</div>
            <button onClick={()=> onPick?.(p.title)} className="mt-4 w-full px-4 py-2 rounded-xl border border-white/15 hover:border-white/30">Выбрать</button>
          </div>
        ))}
      </div>
    </ModalBase>
  );
}

function AddCardModal({ open, onClose }:{ open:boolean; onClose:()=>void }){
  if(!open) return null;
  return (
    <ModalBase onClose={onClose}>
      <h3 className="text-xl font-bold mb-3">Добавить карту</h3>
      <form className="space-y-3" onSubmit={(e)=>{ e.preventDefault(); onClose(); }}>
        <input required placeholder="Номер карты" className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
        <div className="grid grid-cols-2 gap-3">
          <input required placeholder="MM / YY" className="rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
          <input required placeholder="CVC" className="rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"/>
        </div>
        <button className="w-full rounded-xl px-4 py-2 font-semibold btn-main">Оплатить и активировать</button>
      </form>
      <p className="text-xs text-gray-400 mt-3">Данные шифруются и не сохраняются на наших серверах.</p>
    </ModalBase>
  );
}

function InvoicesModal({ open, onClose, invoices }:{ open:boolean; onClose:()=>void; invoices: ReturnType<typeof sampleInvoices> }){
  if(!open) return null;
  return (
    <ModalBase onClose={onClose}>
      <h3 className="text-xl font-bold mb-3">Платёжные документы</h3>
      <div className="card rounded-2xl p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="text-left text-gray-300/85">
            <tr>
              <th className="py-2 px-4">Дата</th>
              <th className="py-2 px-4">Описание</th>
              <th className="py-2 px-4">Сумма</th>
              <th className="py-2 px-4">Статус</th>
              <th className="py-2 px-4 text-right">Файл</th>
            </tr>
          </thead>
          <tbody>
            {invoices.map((inv)=> (
              <tr key={inv.id} className="border-t border-white/10">
                <td className="py-3 px-4">{formatDate(inv.dateISO)}</td>
                <td className="py-3 px-4">{inv.title}</td>
                <td className="py-3 px-4">{inv.amount}</td>
                <td className="py-3 px-4">
                  {inv.status === 'paid' ? (
                    <span className="inline-flex items-center gap-1 text-emerald-300/90"><CheckCircle2 className="w-4 h-4"/> Оплачен</span>
                  ) : inv.status === 'pending' ? (
                    <span className="inline-flex items-center gap-1 text-amber-300/90"><Clock className="w-4 h-4"/> Ожидает</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-rose-300/90"><AlertCircle className="w-4 h-4"/> Ошибка</span>
                  )}
                </td>
                <td className="py-3 px-4 text-right"><button className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-white/15 hover:border-white/30"><Download className="w-4 h-4"/> PDF</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ModalBase>
  );
}

// ---- Utils & samples ----
function addDaysISO(days:number){ const d=new Date(); d.setDate(d.getDate()+days); return d.toISOString(); }
function formatDate(iso:string){ const d=new Date(iso); return d.toLocaleDateString('ru-RU', { day:'2-digit', month:'2-digit', year:'numeric' }); }
function formatDateTime(iso:string){ const d=new Date(iso); return d.toLocaleString('ru-RU', { day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' }); }
function sampleUploads(): UploadCardData[]{ const now=Date.now(); return [ { id:'u1', name:'Brand_15sec.mp4', duration:'00:15', createdISO:new Date(now-36e5).toISOString(), status:'ready' }, { id:'u2', name:'Promo_edit_v3.mov', duration:'00:30', createdISO:new Date(now-86e6).toISOString(), status:'processing' }, { id:'u3', name:'Story_long_01.mp4', duration:'01:00', createdISO:new Date(now-172e6).toISOString(), status:'queued' }, ]; }
function sampleInvoices(){ return [ { id:'inv1', dateISO:addDaysISO(-30), title:'Подписка «Универсальный» — период 30 дней', amount:'10 000 ₽', status:'paid' }, { id:'inv2', dateISO:addDaysISO(-60), title:'Подписка «Универсальный» — период 30 дней', amount:'10 000 ₽', status:'paid' }, { id:'inv3', dateISO:addDaysISO(-90), title:'Подписка «Универсальный» — период 30 дней', amount:'10 000 ₽', status:'paid' }, ]; }
