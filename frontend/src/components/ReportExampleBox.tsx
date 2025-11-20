// frontend/src/components/ReportExampleBox.tsx
import React from "react";

type Props = {
  /** Путь до PDF с примером отчёта. По умолчанию: /report_example.pdf (файл в frontend/public) */
  pdfUrl?: string;
  /** Текст на кнопке скачивания */
  ctaText?: string;
  /** Заголовок блока */
  title?: string;
  /** Подзаголовок/описание */
  description?: string;
};

export default function ReportExampleBox({
  pdfUrl = "/report_example.pdf",
  ctaText = "Скачать пример отчёта (PDF)",
  title = "Как выглядит отчёт?",
  description = "После анализа вы получите структурированный PDF-отчёт с метриками внимания, вовлечённости и запоминаемости, а также рекомендациями по улучшению креатива.",
}: Props) {
  return (
    <div className="card rounded-3xl p-6 md:p-7 grid md:grid-cols-2 gap-6 items-center">
      {/* Левая колонка: текст + кнопка */}
      <div>
        <h3 className="text-2xl md:text-3xl font-bold mb-3">{title}</h3>
        <p className="text-sm md:text-base text-gray-300/90 mb-4 max-w-md">
          {description}
        </p>

        <a
          href={pdfUrl}
          download
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 rounded-2xl px-4 py-2 border border-white/65 hover:border-white transition-colors text-sm font-medium"
          aria-label="Скачать пример отчёта в PDF"
        >
          <span
            aria-hidden
            className="inline-flex items-center justify-center w-5 h-5 rounded bg-white/80 text-black text-[10px] font-bold"
          >
            PDF
          </span>
          {ctaText}
        </a>

        <p className="text-[12px] opacity-70 mt-2">
          Помести файл по пути <code className="opacity-90">frontend/public/report_example.pdf</code>, чтобы ссылка работала по умолчанию.
        </p>
      </div>

      {/* Правая колонка: мини-превью “страницы отчёта” с графиками */}
      <div className="relative">
        <div className="rounded-2xl border border-white/10 bg-black/30 p-4 md:p-5 shadow-lg overflow-hidden">
          {/* Шапка файла */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-7 h-7 rounded bg-white/80 text-black text-[11px] font-bold">
                PDF
              </span>
              <div className="text-sm opacity-85">SynapSight Report</div>
            </div>
            <div className="text-[11px] opacity-60">v1.0 • demo</div>
          </div>

          {/* “страница” */}
          <div className="rounded-xl border border-white/10 bg-black/20 p-4">
            <div className="grid grid-cols-3 gap-3">
              {/* Карточка метрики */}
              <MetricCard label="Внимание" value="Высокое" />
              <MetricCard label="Вовлечённ." value="Средняя" />
              <MetricCard label="Запоминаем." value="Выше ср." />
            </div>

            {/* Линейные графики */}
            <div className="mt-4 h-40 md:h-44">
              <ReportChart />
            </div>

            {/* Небольшая легенда */}
            <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] opacity-75">
              <LegendDot label="Внимание" />
              <LegendDot label="Вовлечённость" variant="b" />
              <span className="opacity-60">·</span>
              <span>Кадры: 0:00–0:30</span>
            </div>
          </div>
        </div>

        {/* Отблеск */}
        <div className="pointer-events-none absolute -inset-8 -z-10 opacity-20 blur-3xl"
             style={{ backgroundImage: "var(--btn-grad)" }} />
      </div>

      {/* Локальные стили (без глобальных зависимостей) */}
      <style>{`
        :root { --btn-grad: linear-gradient(90deg,#541BFF,#DB156E); }
        @keyframes reportLineDraw { from { stroke-dashoffset: 1000; } to { stroke-dashoffset: 0; } }
        .report-line { stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: reportLineDraw 1s ease-out forwards; }
        .report-line.delay { animation-delay: .18s; }
      `}</style>
    </div>
  );
}

/* --- Вспомогательные подкомпоненты --- */

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/30 p-3">
      <div className="text-[11px] opacity-70">{label}</div>
      <div className="text-sm font-semibold mt-0.5">{value}</div>
    </div>
  );
}

function LegendDot({ label, variant }: { label: string; variant?: "a" | "b" }) {
  const color = variant === "b" ? "#c44fb0" : "#4f8bff";
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        aria-hidden
        className="inline-block w-2.5 h-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}

function ReportChart() {
  return (
    <svg viewBox="0 0 320 180" className="w-full h-full text-gray-400/90" aria-hidden="true">
      {/* Фоновая сетка */}
      <g stroke="currentColor" strokeWidth="0.4" className="opacity-30">
        {/* горизонтальные */}
        {[0, 30, 60, 90, 120, 150, 180].map((y) => (
          <line key={`h-${y}`} x1="0" x2="320" y1={y} y2={y} />
        ))}
        {/* вертикальные */}
        {[0, 64, 128, 192, 256, 320].map((x) => (
          <line key={`v-${x}`} y1="0" y2="180" x1={x} x2={x} />
        ))}
      </g>

      {/* Две кривые */}
      <path
        d="M 0 130 Q 20 120, 40 110 T 80 130 T 120 120 T 160 135 T 200 125 T 240 140 T 280 130 T 320 135"
        fill="none"
        stroke="#4f8bff"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="report-line"
      />
      <path
        d="M 0 125 Q 20 105, 40 95 T 80 120 T 120 115 T 160 130 T 200 120 T 240 135 T 280 120 T 320 130"
        fill="none"
        stroke="#c44fb0"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="report-line delay"
      />
    </svg>
  );
}
