// frontend/src/pages/AccountDashboard.tsx
import React, { useEffect, useState } from "react";
import {
  LogOut, CreditCard, FileText, CalendarClock, X, CheckCircle2, User, Mail,
  Lock, Briefcase, Eye, EyeOff, ShieldCheck, Timer, Info,
  UploadCloud, BarChart3
} from "lucide-react";
import { useNavigate } from 'react-router-dom';

// Единые шапка и футер
import SiteHeader from "../components/SiteHeader";
import SiteFooter from "../components/SiteFooter";

/** Полный кабинет с модалками профиля/почты/пароля и покупкой минут */
export default function AccountDashboard({ onLogout }: { onLogout: () => void }) {
  const [user, setUser] = useState({
    name: "Анна Маркетолог",
    email: "anna@synapsight.io",
    occupation: "Маркетолог",
    plan: { title: "Универсальный", price: "10 000 ₽ / мес" },
    billing: { nextChargeISO: addDaysISO(11) },
    usage: { usedMin: 18, quotaMin: 60 },
  });
  const usagePct = Math.min(100, Math.round((user.usage.usedMin / user.usage.quotaMin) * 100));

  const [showBuy, setShowBuy] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [lastPurchase, setLastPurchase] = useState<{ minutes: number; total: number; method: string } | null>(null);

  const [showProfile, setShowProfile] = useState(false);
  const [showEmailInfo, setShowEmailInfo] = useState(false);
  const [showEmailChange, setShowEmailChange] = useState(false);
  const [showEmailVerify, setShowEmailVerify] = useState(false);
  const [pendingEmail, setPendingEmail] = useState<string | null>(null);
  const [showEmailSuccess, setShowEmailSuccess] = useState(false);

  const [showPasswordChange, setShowPasswordChange] = useState(false);
  const openPasswordChange = () => setShowPasswordChange(true);

  return (
    <div className="relative min-h-screen overflow-x-clip bg-gradient-to-b from-[#0B0B12] via-[#171727] to-[#0B0B12] text-white">
      <style>{`
        .card{background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01));border:1px solid rgba(255,255,255,.08);backdrop-filter:blur(6px)}
        :root{--btn-grad:linear-gradient(90deg,#541BFF,#DB156E)}
        @keyframes popIn{0%{transform:scale(.7);opacity:0}60%{transform:scale(1.05);opacity:1}100%{transform:scale(1)}}
      `}</style>

      {/* ЕДИНЫЙ HEADER */}
      <SiteHeader
        nav={[]}
        rightSlot={
          <>
            <button
              onClick={() => setShowProfile(true)}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/15 hover:border-white/30 max-w-[220px] truncate text-sm"
            >
              <User className="w-4 h-4" /> {user.name}
            </button>
            <button
              onClick={onLogout}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/15 hover:border-white/30 text-sm"
            >
              <LogOut className="w-4 h-4" /> Выйти
            </button>
          </>
        }
      />

      <main className="mx-auto max-w-7xl px-5 py-8 space-y-8">
        <button
          onClick={onLogout}
          className="inline-flex items-center gap-2 px-3 py-1.5 mb-4 rounded-xl border border-white/15 hover:border-white/30 text-sm"
        >
          ← Вернуться на главную
        </button>

        {/* Верхняя карточка: план/использование */}
        <section className="grid lg:grid-cols-1 gap-6">
          <div className="bg-gradient-to-b from-[#111119] to-[#171727] border border-white/10 rounded-2xl p-5 shadow-lg shadow-black/40 relative">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="min-w-[260px]">
                <h1 className="text-2xl font-bold">Личный кабинет</h1>
                <p className="text-sm text-gray-300/85">Здравствуйте, {user.name.split(' ')[0]}! Здесь ваш план и использование минут.</p>
              </div>
              <div className="flex items-center gap-2">
                <button className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/15 hover:border-white/30 text-sm">
                  <CreditCard className="w-4 h-4" /> Карта
                </button>
                <button className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-white/15 hover:border-white/30 text-sm">
                  <FileText className="w-4 h-4" /> Платёжные документы
                </button>
              </div>
            </div>

            <div className="mt-6 grid md:grid-cols-3 gap-4">
              <div className="relative p-4 rounded-xl border border-white/10 bg-gradient-to-b from-[#1C1C2A] to-[#111119]">
                <div className="absolute top-3.right-3">
                  <button className="text-xs px-2 py-1 rounded-lg border border-white/15 hover:border-white/30">Изменить тариф</button>
                </div>
                <div className="text-xs opacity-80">План</div>
                <div className="text-lg font-semibold mt-1">{user.plan.title}</div>
                <div className="text-xs opacity-70 mt-0.5">{user.plan.price}</div>
                <div className="mt-2 text-xs opacity-80 inline-flex items-center gap-1">
                  <CalendarClock className="w-3.5 h-3.5" /> Следующее списание: {formatDate(user.billing.nextChargeISO)}
                </div>
              </div>

              <div className="relative p-4 rounded-xl border border-white/10 bg-gradient-to-b from-[#1C1C2A] to-[#111119] md:col-span-2">
                <div>
                  <div className="text-xs opacity-80 mb-2">Использование минут</div>
                  <UsageBar pct={usagePct} />
                  <div className="mt-2 text-xs opacity-70">{user.usage.usedMin} из {user.usage.quotaMin} мин</div>
                </div>
                <div className="absolute bottom-3 right-3">
                  <button
                    onClick={() => setShowBuy(true)}
                    className="text-xs px-3 py-2 rounded-lg border border-white/15 hover:border-white/30 bg-[rgba(84,27,255,0.2)] backdrop-blur-md"
                  >
                    Добавить минуты
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Отдельный блок: Загрузка + История из БД */}
        <section>
          <AccountUploadsAndHistory />
        </section>
      </main>

      {/* ЕДИНЫЙ FOOTER */}
      <SiteFooter />

      {/* Покупка минут / успешная оплата */}
      <BuyMinutesModal
        open={showBuy}
        onClose={() => setShowBuy(false)}
        boundCardLabel="VISA •• 1234"
        onPaid={({ minutes, total, method }) => { setShowBuy(false); setLastPurchase({ minutes, total, method }); setShowSuccess(true); }}
      />
      <PaymentSuccessModal open={showSuccess} onClose={() => setShowSuccess(false)} details={lastPurchase} />

      {/* Профиль + почта + пароль */}
      <EmailSuccessModal open={showEmailSuccess} onClose={() => setShowEmailSuccess(false)} />

      <ProfileModal
        open={showProfile}
        onClose={() => setShowProfile(false)}
        user={user}
        onStartEmailInfo={() => { setShowProfile(false); setShowEmailInfo(true); }}
        onOpenPasswordChange={openPasswordChange}
        onSaveBasic={(payload) => { setUser(u => ({ ...u, ...payload })); setShowProfile(false); }}
      />

      <EmailInfoModal
        open={showEmailInfo}
        masked={maskEmail(user.email)}
        onClose={() => setShowEmailInfo(false)}
        onChange={() => { setShowEmailInfo(false); setShowEmailChange(true); }}
      />

      <EmailChangeModal
        open={showEmailChange}
        currentMasked={maskEmail(user.email)}
        onClose={() => setShowEmailChange(false)}
        onSubmit={(newEmail) => { setPendingEmail(newEmail); setShowEmailChange(false); setShowEmailVerify(true); }}
      />

      <EmailVerifyModal
        open={showEmailVerify}
        email={pendingEmail || ''}
        onClose={() => { setShowEmailVerify(false); setPendingEmail(null); }}
        onVerified={() => { if (pendingEmail) { setUser(u => ({ ...u, email: pendingEmail })); } setShowEmailVerify(false); setPendingEmail(null); setShowEmailSuccess(true); }}
      />

      <PasswordChangeModal open={showPasswordChange} onClose={() => setShowPasswordChange(false)} />
    </div>
  );
}

/* ---------- Модалки и утилиты ---------- */

function BuyMinutesModal({
  open, onClose, boundCardLabel, onPaid,
}: { open: boolean; onClose: () => void; boundCardLabel: string; onPaid: (p: { minutes: number; total: number; method: string }) => void; }) {
  const PRICE_PER_MIN = 150;
  const [minutes, setMinutes] = useState<number>(5);
  const [method, setMethod] = useState<'bound' | 'new' | 'sbp'>('bound');
  if (!open) return null;
  const total = Math.max(0, Math.floor((minutes || 0))) * PRICE_PER_MIN;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.94)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4" /></button>
        <h3 className="text-xl font-bold.mb-1">Добавить минуты</h3>
        <div className="text-xs opacity-80 mb-4">1 минута = 150 ₽</div>

        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); onPaid({ minutes: Math.max(1, minutes || 1), total, method }); }}>
          <label className="block text-sm">
            <span className="opacity-80">Сколько минут покупаем</span>
            <input
              type="number" min={1} step={1} value={minutes}
              onChange={(e) => setMinutes(Math.max(1, Number(e.target.value || 1)))}
              className="mt-1 w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30"
            />
          </label>

          <label className="block text-sm">
            <span className="opacity-80">Способ оплаты</span>
            <select
              value={method} onChange={(e) => setMethod(e.target.value as any)}
              className="mt-1 w-full rounded-xl bg-white/10 border border-white/15 px-3 py-2 outline-none focus:border-white/30 appearance-none"
            >
              <option value="bound" className="bg-[#171727]">Карта — {boundCardLabel}</option>
              <option value="new" className="bg-[#171727]">Новая карта</option>
              <option value="sbp" className="bg-[#171727]">СБП (Система быстрых платежей)</option>
            </select>
          </label>

          {method === 'new' && (
            <div className="space-y-3">
              <input placeholder="Номер карты" className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30" />
              <div className="grid grid-cols-2 gap-3">
                <input placeholder="MM / YY" className="rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30" />
                <input placeholder="CVC" className="rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30" />
              </div>
            </div>
          )}

          {method === 'sbp' && (
            <p className="text-xs opacity-80">После подтверждения мы покажем QR-код/ссылку для оплаты через СБП.</p>
          )}

          <div className="flex items-center justify-between pt-2">
            <div className="text-sm opacity-85">Итого: <span className="font-semibold">{total.toLocaleString('ru-RU')} ₽</span></div>
            <button className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Оплатить</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function PaymentSuccessModal({ open, onClose, details }: { open: boolean; onClose: () => void; details: { minutes: number; total: number; method: string } | null; }) {
  useEffect(() => { if (!open) return; const t = setTimeout(onClose, 1600); return () => clearTimeout(t); }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-[360px] mx-4 card rounded-2xl p-6 bg-[rgba(21,19,29,0.94)] border-white/20 text-center" style={{ animation: 'popIn 400ms ease-out' }}>
        <div className="mx-auto w-16 h-16 rounded-full flex items-center justify-center" style={{ backgroundImage: 'var(--btn-grad)' }}>
          <CheckCircle2 className="w-10 h-10" />
        </div>
        <div className="mt-3 text-lg font-semibold">Оплата прошла успешно</div>
        {details && <div className="mt-1 text-sm opacity-80">Добавлено {details.minutes} мин · {details.total.toLocaleString('ru-RU')} ₽</div>}
        <button onClick={onClose} className="mt-4 inline-flex items-center justify-center rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Ок</button>
      </div>
    </div>
  );
}

function EmailSuccessModal({ open, onClose }: { open: boolean; onClose: () => void; }) {
  useEffect(() => { if (!open) return; const t = setTimeout(onClose, 1400); return () => clearTimeout(t); }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-[360px] mx-4 card rounded-2xl p-6 bg-[rgba(21,19,29,0.94)] border-white/20 text-center" style={{ animation: 'popIn 400ms.ease-out' }}>
        <div className="mx-auto w-16 h-16 rounded-full flex items-center justify-center" style={{ backgroundImage: 'var(--btn-grad)' }}>
          <CheckCircle2 className="w-10 h-10" />
        </div>
        <div className="mt-3 text-lg font-semibold">Почта успешно обновлена</div>
        <button onClick={onClose} className="mt-4 inline-flex items-center justify-center rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Ок</button>
      </div>
    </div>
  );
}

function EmailInfoModal({ open, masked, onClose, onChange }: { open: boolean; masked: string; onClose: () => void; onChange: () => void; }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.95)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4" /></button>
        <div className="flex items-center gap-2 text-sm opacity-80 mb-1"><Info className="w-4 h-4" /> Почта</div>
        <div className="text-xl font-bold mb-2">{masked}</div>
        <p className="text-sm opacity-85">Этот адрес почты будет использоваться при каждом входе в аккаунт.</p>
        <div className="mt-4 flex items-center justify-end gap-2">
          <button onClick={onClose} className="rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Отмена</button>
          <button onClick={onChange} className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Изменить</button>
        </div>
      </div>
    </div>
  );
}

function EmailChangeModal({
  open, currentMasked, onClose, onSubmit,
}: { open: boolean; currentMasked: string; onClose: () => void; onSubmit: (email: string) => void; }) {
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { if (open) { setEmail(""); setError(null); } }, [open]);
  if (!open) return null;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.includes('@')) { setError('Введите корректный e-mail'); return; }
    onSubmit(email);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.95)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4" /></button>
        <div className="text-sm opacity-80 mb-1">Текущая почта</div>
        <div className="text-base mb-3">{currentMasked}</div>
        <h3 className="text-xl font-bold mb-2">Новая почта</h3>
        <form className="space-y-3" onSubmit={submit}>
          <input autoFocus type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com"
                 className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30" />
          {error && <div className="text[13px] text-red-300">{error}</div>}
          <div className="flex items-center justify-end gap-2">
            <button type="button" onClick={onClose} className="rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Отмена</button>
            <button type="submit" className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Продолжить</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function EmailVerifyModal({
  open, email, onClose, onVerified,
}: { open: boolean; email: string; onClose: () => void; onVerified: () => void; }) {
  const [code, setCode] = useState("");
  const [timer, setTimer] = useState(60);
  useEffect(() => {
    if (!open) return;
    setCode(""); setTimer(60);
    const iv = setInterval(() => setTimer(t => t > 0 ? t - 1 : 0), 1000);
    return () => clearInterval(iv);
  }, [open]);
  if (!open) return null;

  const canResend = timer === 0;
  const handleVerify = (e: React.FormEvent) => { e.preventDefault(); if (code.trim().length >= 4) { onVerified(); } };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.95)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4" /></button>
        <div className="flex items-center gap-2 text-sm opacity-80 mb-1"><ShieldCheck className="w-4 h-4" /> Подтверждение почты</div>
        <h3 className="text-xl font-bold mb-2">Мы отправили код на {email}</h3>
        <form className="space-y-4" onSubmit={handleVerify}>
          <input autoFocus value={code} onChange={(e) => setCode(e.target.value)} placeholder="Код из письма"
                 className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30 tracking-widest" />
          <div className="flex items-center justify-between text-xs opacity-80">
            <div className="inline-flex items-center gap-1"><Timer className="w-3.5 h-3.5" /> Повторить отправку {canResend ? 'можно' : `через ${timer}с`}</div>
            <button type="button" disabled={!canResend} className="underline decoration-dotted disabled:opacity-40" onClick={() => setTimer(60)}>Отправить код ещё раз</button>
          </div>
          <div className="flex items-center justify-end gap-2">
            <button type="button" onClick={onClose} className="rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Отмена</button>
            <button type="submit" className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Подтвердить</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function PasswordChangeModal({ open, onClose }: { open: boolean; onClose: () => void; }) {
  const [step, setStep] = useState<'current' | 'new' | 'done'>('current');
  const [current, setCurrent] = useState('');
  const [newPwd, setNewPwd] = useState('');
  const [newPwd2, setNewPwd2] = useState('');
  const [show1, setShow1] = useState(false);
  const [show2, setShow2] = useState(false);

  useEffect(() => { if (open) { setStep('current'); setCurrent(''); setNewPwd(''); setNewPwd2(''); setShow1(false); setShow2(false); } }, [open]);
  if (!open) return null;

  const submitCurrent = (e: React.FormEvent) => { e.preventDefault(); if (current.trim().length >= 4) { setStep('new'); } };
  const submitNew = (e: React.FormEvent) => {
    e.preventDefault();
    if (newPwd.length < 6) return;
    if (newPwd !== newPwd2) return;
    setStep('done');
    setTimeout(() => { onClose(); }, 1200);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.95)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg.white/10" onClick={onClose}><X className="w-4 h-4" /></button>
        <div className="flex items-center gap-2 text-sm opacity-80 mb-1"><Lock className="w-4 h-4" /> Смена пароля</div>

        {step === 'current' && (
          <>
            <h3 className="text-xl font-bold mb-2">Введите текущий пароль</h3>
            <form className="space-y-3" onSubmit={submitCurrent}>
              <div className="relative">
                <input type={show1 ? 'text' : 'password'} value={current} onChange={(e) => setCurrent(e.target.value)}
                       className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 pr-9 outline-none focus:border-white/30" placeholder="Текущий пароль" />
                <button type="button" onClick={() => setShow1(v => !v)} className="absolute right-2 top-1/2 -translate-y-1/2 opacity-80 hover:opacity-100">
                  {show1 ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <div className="flex items-center justify-end gap-2">
                <button type="button" onClick={onClose} className="rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Отмена</button>
                <button type="submit" className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Далее</button>
              </div>
            </form>
          </>
        )}

        {step === 'new' && (
          <>
            <h3 className="text-xl font-bold mb-2">Новый пароль</h3>
            <form className="space-y-3" onSubmit={submitNew}>
              <div className="relative">
                <input type={show1 ? 'text' : 'password'} value={newPwd} onChange={(e) => setNewPwd(e.target.value)}
                       className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 pr-9 outline-none focus:border-white/30" placeholder="Новый пароль (мин. 6 символов)" />
                <button type="button" onClick={() => setShow1(v => !v)} className="absolute right-2 top-1/2 -translate-y-1/2 opacity-80 hover:opacity-100">
                  {show1 ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <div className="relative">
                <input type={show2 ? 'text' : 'password'} value={newPwd2} onChange={(e) => setNewPwd2(e.target.value)}
                       className="w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 pr-9 outline-none focus:border-white/30" placeholder="Повторите новый пароль" />
                <button type="button" onClick={() => setShow2(v => !v)} className="absolute right-2 top-1/2 -translate-y-1/2 opacity-80 hover:opacity-100">
                  {show2 ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <div className="flex items-center justify-between text-xs opacity-75"><span>Минимум 6 символов</span></div>
              <div className="flex items-center justify-end gap-2">
                <button type="button" onClick={() => setStep('current')} className="rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Назад</button>
                <button type="submit" className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Сохранить</button>
              </div>
            </form>
          </>
        )}

        {step === 'done' && (
          <div className="text-center" style={{ animation: 'popIn 350ms ease-out' }}>
            <div className="mx-auto w-16 h-16 rounded-full flex items-center justify-center" style={{ backgroundImage: 'var(--btn-grad)' }}>
              <CheckCircle2 className="w-10 h-10" />
            </div>
            <div className="mt-3 text-lg font-semibold">Пароль изменён</div>
          </div>
        )}
      </div>
    </div>
  );
}

function ProfileModal({
  open, onClose, user, onStartEmailInfo, onOpenPasswordChange, onSaveBasic,
}: {
  open: boolean; onClose: () => void; user: any;
  onStartEmailInfo: () => void; onOpenPasswordChange: () => void; onSaveBasic: (p: { name: string; occupation: string }) => void;
}) {
  const [name, setName] = useState(user.name);
  const [occupation, setOccupation] = useState(user.occupation || "Маркетолог");

  useEffect(() => { if (open) { setName(user.name); setOccupation(user.occupation); } }, [open, user]);
  if (!open) return null;

  const submit = (e: React.FormEvent) => { e.preventDefault(); onSaveBasic({ name, occupation }); };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center text-white">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md card rounded-2xl p-6 bg-[rgba(21,19,29,0.94)] border-white/20">
        <button className="absolute top-2 right-2 p-2 rounded-lg hover:bg-white/10" onClick={onClose}><X className="w-4 h-4" /></button>
        <h3 className="text-xl font-bold mb-4">Настройки профиля</h3>
        <form className="space-y-4" onSubmit={submit}>
          <label className="block text-sm">
            <span className="opacity-80 inline-flex items-center gap-2"><User className="w-4 h-4" /> Имя</span>
            <input value={name} onChange={(e) => setName(e.target.value)} className="mt-1 w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2 outline-none focus:border-white/30" />
          </label>

          <div className="block text-sm">
            <div className="opacity-80 inline-flex items-center gap-2 mb-1"><Mail className="w-4 h-4" /> Почта</div>
            <div className="flex items-center justify-between w-full rounded-xl bg-white/5 border border-white/15 px-3 py-2">
              <div className="text-sm opacity-90 select-text">{maskEmail(user.email)}</div>
              <button type="button" onClick={onStartEmailInfo} className="text-sm underline underline-offset-4 opacity-90 hover:opacity-100">Подробнее</button>
            </div>
          </div>

          <label className="block text-sm">
            <span className="opacity-80 inline-flex items-center gap-2"><Briefcase className="w-4 h-4" /> Род деятельности</span>
            <select value={occupation} onChange={(e) => setOccupation(e.target.value)}
                    className="mt-1 w-full rounded-xl bg-white/10 border border-white/15 px-3 py-2 outline-none focus:border-white/30">
              <option className="bg-[#171727]" value="Маркетолог">Маркетолог</option>
              <option className="bg-[#171727]" value="Продюсер">Продюсер</option>
              <option className="bg-[#171727]" value="Видеомонтаж">Видеомонтаж</option>
              <option className="bg-[#171727]" value="Аналитик">Аналитик</option>
              <option className="bg-[#171727]" value="Другое">Другое</option>
            </select>
          </label>

          <div className="block text-sm">
            <button
              type="button"
              onClick={onOpenPasswordChange}
              className="inline-flex items-center gap-2 rounded-xl bg-white/5 border border-white/15 px-3 py-2 hover:bg-white/10 transition-colors"
            >
              <Lock className="w-4 h-4" /> Смена пароля
            </button>
          </div>

          <div className="flex items-center justify-end gap-2 pt-1">
            <button type="button" onClick={onClose} className="rounded-xl px-4 py-2 border border-white/20 hover:border-white/35">Отмена</button>
            <button type="submit" className="rounded-xl px-4 py-2 font-semibold" style={{ backgroundImage: 'var(--btn-grad)', color: '#fff' }}>Сохранить</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function UsageBar({ pct }: { pct: number }) {
  return (
    <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
      <div className="h-full" style={{ width: `${pct}%`, backgroundImage: 'var(--btn-grad)' }} />
    </div>
  );
}

function maskEmail(email: string) {
  const [local, domain] = String(email).split('@');
  if (!local || !domain) return String(email).toUpperCase();
  const show = Math.min(6, Math.max(2, Math.floor(local.length / 2)));
  const maskedLocal = `${'*'.repeat(Math.max(0, local.length - show))}${local.slice(-show).toUpperCase()}`;
  return `${maskedLocal}@${domain.toUpperCase()}`;
}

function addDaysISO(days: number) { const d = new Date(); d.setDate(d.getDate() + days); return d.toISOString(); }
function formatDate(iso: string) { const d = new Date(iso); return d.toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' }); }

/* ---------- Общие типы (один раз) ---------- */
type VideoStatus = 'UPLOADING'|'QUEUED'|'PROCESSING'|'FAILED'|'DONE';
type VideoItem = {
  id: string;
  filename: string;
  storage_key?: string;
  created_at?: string;
  duration_sec?: number | null;
  status: VideoStatus;
  progress?: number | null;
};

function StatusBadge({ s }: { s: VideoItem['status'] }) {
  const map = {
    DONE:   'bg-green-500/20 text-green-300 border-green-500/30',
    FAILED: 'bg-red-500/20 text-red-300 border-red-500/30',
    PROCESSING: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    QUEUED: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
    UPLOADING: 'bg-gray-500/20 text-gray-300 border-gray-500/30',
  } as const;
  return <span className={`text-[11px] px-2 py-0.5 rounded-lg border ${map[s]}`}>{s}</span>;
}

/* ========= Account: Upload + History (DB-backed) ========= */

export function AccountUploadsAndHistory() {
  const [items, setItems] = React.useState<VideoItem[] | null>(null);
  const [busy, setBusy] = React.useState(false);

  // первичная загрузка истории
  React.useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/videos?limit=50', { credentials: 'include' });
        const data = await r.json();
        const list: VideoItem[] = Array.isArray(data) ? data : (data.items ?? []);
        list.sort((a,b) => (new Date(b.created_at||0).getTime() - new Date(a.created_at||0).getTime()));
        setItems(list);
      } catch (e) {
        console.error('videos list failed', e);
        setItems([]);
      }
    })();
  }, []);

  // онлайн-обновления через SSE
  React.useEffect(() => {
    const es = new EventSource('/api/events', { withCredentials: true } as any);
    es.onmessage = (e) => {
      try {
        const m = JSON.parse(e.data); // {event, data}
        if (!m?.event || !m?.data) return;
        if (m.event === 'video.updated' || m.event === 'job.updated' || m.event === 'job.progress') {
          const vid = m.data.video_id || m.data.id;
          setItems(prev => {
            if (!prev) return prev;
            return prev.map(v => v.id === vid
              ? {
                  ...v,
                  status: (m.data.status ?? v.status),
                  progress: typeof m.data.progress === 'number' ? m.data.progress : v.progress
                }
              : v
            );
          });
        }
        if (m.event === 'video.created' && m.data?.video) {
          setItems(prev => [m.data.video as VideoItem, ...(prev ?? [])]);
        }
      } catch {}
    };
    return () => es.close();
  }, []);

  return (
    <section className="card rounded-3xl p-8 space-y-6">
      <UploadAreaDashConnected
        disabled={busy}
        onBusy={setBusy}
        onUploaded={(video) => { setItems(prev => [video, ...(prev ?? [])]); }}
      />

      {/* История роликов */}
      <div className="space-y-3">
        <div className="text-sm opacity-80">История роликов</div>
        {items === null ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({length:6}).map((_,i)=>(
              <div key={i} className="h-24 rounded-xl border border-white/10 bg-white/5 animate-pulse" />
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 text-sm opacity-80">
            Пока пусто. Загрузите видео — после постановки в очередь оно появится здесь.
          </div>
        ) : (
          <div className="space-y-3">
            {items.map(v => (<VideoRow key={v.id} v={v} />))}
          </div>
        )}
      </div>
    </section>
  );
}

/* ---- upload area (uses /api/uploads/init → S3 POST → /api/uploads/complete) ---- */

function UploadAreaDashConnected({
  disabled,
  onBusy,
  onUploaded
}: {
  disabled?: boolean
  onBusy?: (b: boolean) => void
  onUploaded: (v: VideoItem) => void
}) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = React.useState(false);
  const [queue, setQueue] = React.useState<{id:string; name:string; progress:number; localFile?: File}[]>([]);
  const timers = React.useRef<Record<string, any>>({});

  React.useEffect(() => () => { Object.values(timers.current).forEach(clearInterval); }, []);

  const onPick = () => inputRef.current?.click();
  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => { handleFiles(e.target.files); (e.target as any).value = ''; };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files); };

  const handleFiles = (fileList: FileList | null) => {
    const files = Array.from(fileList ?? []);
    files.forEach(file => startUploadFlow(file));
  };

  async function startUploadFlow(file: File) {
    const tempId = `${Date.now()}_${Math.random().toString(36).slice(2,7)}`;
    setQueue(prev => [...prev, { id: tempId, name: file.name, progress: 0, localFile: file }]);

    onBusy?.(true);
    let initResp: { upload_url: string; fields: Record<string,string>; key: string; video_id: string };
    try {
      const r = await fetch('/api/uploads/init', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, size: file.size, mime: file.type || 'video/mp4' })
      });
      if (!r.ok) throw new Error(`init ${r.status}`);
      initResp = await r.json();
    } catch (e) {
      console.error('init failed', e);
      failTemp(tempId, 'Инициализация загрузки не удалась');
      onBusy?.(false);
      return;
    }

    try {
      await uploadToPresignedPost(initResp.upload_url, initResp.fields, file, (p) => {
        setQueue(prev => prev.map(q => q.id === tempId ? ({ ...q, progress: Math.max(q.progress, p) }) : q));
      });
    } catch (e) {
      console.error('upload failed', e);
      failTemp(tempId, 'Ошибка загрузки в хранилище');
      onBusy?.(false);
      return;
    }

    try {
      const r2 = await fetch('/api/uploads/complete', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: initResp.video_id,
          key: initResp.key,
          filename: file.name,
          size: file.size,
          mime: file.type || 'video/mp4'
        })
      });
      if (!r2.ok) throw new Error(`complete ${r2.status}`);
      const data = await r2.json();
      const video: VideoItem = data.video ?? data;
      setQueue(prev => prev.filter(q => q.id !== tempId));
      onUploaded(video);
    } catch (e) {
      console.error('complete failed', e);
      failTemp(tempId, 'Ошибка завершения загрузки');
    } finally {
      onBusy?.(false);
    }
  }

  function failTemp(id: string, _msg?: string) {
    setQueue(prev => prev.map(q => q.id === id ? ({ ...q, progress: 0 }) : q));
  }

  return (
    <div>
      <div
        onDragOver={(e)=>{ e.preventDefault(); setDrag(true); }}
        onDragLeave={()=> setDrag(false)}
        onDrop={onDrop}
        onClick={disabled ? undefined : onPick}
        className={`text-center border-2 border-dashed rounded-xl p-10 transition 
          ${drag? 'border-white/60 bg-white/10':'border-white/20 hover:border-white/40 hover:bg-white/5'}
          ${disabled ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}
        `}
        role="button" aria-label="Выбрать файл"
      >
        <UploadCloud className="w-10 h-10 mx-auto mb-2" />
        <p className="text-sm text-gray-300/85">Перетащите видео или выберите файл для анализа (.mp4 / .mov)</p>
        <button type="button" disabled={disabled} className="mt-4 inline-flex items-center gap-2 px-4 py-2 border border-white/20 rounded-xl hover:border-white/40 disabled:opacity-60">
          <UploadCloud className="w-4 h-4" /> Выбрать файл
        </button>
        <input ref={inputRef} type="file" className="hidden" accept="video/mp4,video/quicktime" multiple onChange={onChange}/>
      </div>

      {queue.length>0 && (
        <div className="mt-4 space-y-3">
          {queue.map(q => (
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

async function uploadToPresignedPost(
  uploadUrl: string,
  fields: Record<string,string>,
  file: File,
  onProgress?: (pct: number) => void
) {
  const form = new FormData();
  Object.entries(fields).forEach(([k,v]) => form.append(k, v));
  form.append('file', file);

  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', uploadUrl, true);
    xhr.upload.onprogress = (e) => {
      if (!onProgress || !e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      onProgress(pct);
    };
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300) ? resolve() : reject(new Error(`POST ${xhr.status}`));
    xhr.onerror = () => reject(new Error('network'));
    xhr.send(form);
  });
}

/* ---- row for history ---- */

function VideoRow({ v }: { v: VideoItem }) {
  const navigate = useNavigate();
  const duration = v.duration_sec ? secToClock(v.duration_sec) : '—:—';
  const created = v.created_at ? new Date(v.created_at).toLocaleString('ru-RU', { day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' }) : '';
  return (
    <div className="rounded-xl border border-white/12 bg-white/5 p-5 flex items-center gap-4">
      <div className="shrink-0 w-14 h-10 rounded bg-white/10 flex items-center justify-center text-[10px] opacity-80">MP4</div>
      <div className="min-w-0 grow">
        <div className="text-base font-medium truncate">{v.filename || `Видео ${v.id.slice(0,8)}`}</div>
        <div className="text-[12px] opacity-60 truncate">{created} • {duration}</div>
        {typeof v.progress === 'number' && v.status !== 'DONE' && v.status !== 'FAILED' && (
          <div className="mt-2">
            <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-[#541BFF] to-[#DB156E]" style={{ width: `${v.progress}%` }} />
            </div>
            <div className="text-[11px] opacity-70 mt-1">{v.progress}%</div>
          </div>
        )}
      </div>
      <StatusBadge s={v.status} />
      <button
        className="ml-auto inline-flex items-center gap-1 text-[13px] px-4 py-2 rounded-lg border border-white/12 bg-white/5 hover:bg-white/10 transition-colors"
        onClick={() => navigate(`/metrics/${v.id}`)}
      >
        <BarChart3 className="w-4 h-4" /> Смотреть метрики
      </button>
    </div>
  );
}

function secToClock(s:number){
  const m = Math.floor(s/60); const ss = s%60|0;
  return `${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}
