"""Generate interactive HTML report with loudness, tempo, and dynamics metrics from a video's audio track.

This script extracts the audio from a video (default: data/video/000-youtube.mp4),
computes loudness-related metrics (LUFS), tempo/percussiveness, and dynamic contrast
statistics, then produces a Russian-language HTML report with detailed explanations
and interactive Plotly visualizations.
"""
from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import ffmpeg
import librosa
import numpy as np
import plotly.graph_objects as go
import pyloudnorm as pyln


@dataclass
class LoudnessMetrics:
    integrated_lufs: float
    lra: float
    true_peak_dbfs: float
    crest_factor_db: float
    short_term_lufs: np.ndarray
    momentary_lufs: np.ndarray
    short_term_times: np.ndarray
    momentary_times: np.ndarray
    hook_loudness_delta: float
    hook_mean_short_term: float
    baseline_mean_short_term: float


@dataclass
class TempoMetrics:
    tempo_bpm: float
    tempo_confidence: float
    onset_density_hook: float
    percussive_ratio_overall: float
    percussive_ratio_hook: float
    tempo_stability: float | None
    beat_times: np.ndarray


@dataclass
class DynamicsMetrics:
    rms_dbfs: np.ndarray
    rms_times: np.ndarray
    transient_punch_hook: float
    transient_punch_body: float


def extract_audio(video_path: Path, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Extract mono audio from a video file using ffmpeg and return the waveform and sample rate."""
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"
        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), ac=1, ar=target_sr, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    return waveform, sr


def compute_loudness_metrics(audio: np.ndarray, sr: int) -> LoudnessMetrics:
    meter = pyln.Meter(sr)  # ITU-R BS.1770 meter with K-weighting

    def _sliding_loudness(window_s: float, hop_s: float) -> Tuple[np.ndarray, np.ndarray]:
        window = int(window_s * sr)
        hop = max(1, int(hop_s * sr))
        if len(audio) == 0:
            return np.array([]), np.array([])

        values: list[float] = []
        centers: list[float] = []

        if len(audio) < window:
            try:
                values.append(float(meter.integrated_loudness(audio)))
            except Exception:
                values.append(float("nan"))
            centers.append(len(audio) / (2 * sr))
        else:
            for start in range(0, len(audio) - window + 1, hop):
                segment = audio[start : start + window]
                try:
                    loudness_value = float(meter.integrated_loudness(segment))
                except Exception:
                    loudness_value = float("nan")
                values.append(loudness_value)
                centers.append((start + window / 2) / sr)

        return np.array(centers), np.array(values)

    integrated_lufs = meter.integrated_loudness(audio)
    true_peak_dbfs = meter.true_peak(audio)

    rms_linear = np.sqrt(np.mean(np.square(audio)))
    rms_dbfs = 20 * np.log10(max(rms_linear, 1e-12))
    crest_factor_db = true_peak_dbfs - rms_dbfs

    short_term_times, short_term_lufs = _sliding_loudness(window_s=3.0, hop_s=0.5)
    momentary_times, momentary_lufs = _sliding_loudness(window_s=0.4, hop_s=0.1)

    finite_short_term = short_term_lufs[np.isfinite(short_term_lufs)]
    finite_short_term = finite_short_term[finite_short_term > -70]
    if finite_short_term.size:
        relative_gate = integrated_lufs - 20
        gated = finite_short_term[finite_short_term > relative_gate]
        if gated.size:
            p10 = np.nanpercentile(gated, 10)
            p95 = np.nanpercentile(gated, 95)
            lra = float(p95 - p10)
        else:
            lra = float("nan")
    else:
        lra = float("nan")

    hook_mask = (short_term_times >= 0) & (short_term_times < 5)
    baseline_mask = (short_term_times >= 5) & (short_term_times < 15)

    hook_mean_short_term = float(np.nanmean(short_term_lufs[hook_mask])) if hook_mask.any() else float("nan")
    baseline_mean_short_term = float(
        np.nanmean(short_term_lufs[baseline_mask])
    ) if baseline_mask.any() else float("nan")
    hook_loudness_delta = hook_mean_short_term - baseline_mean_short_term

    return LoudnessMetrics(
        integrated_lufs=integrated_lufs,
        lra=lra,
        true_peak_dbfs=true_peak_dbfs,
        crest_factor_db=crest_factor_db,
        short_term_lufs=short_term_lufs,
        momentary_lufs=momentary_lufs,
        short_term_times=short_term_times,
        momentary_times=momentary_times,
        hook_loudness_delta=hook_loudness_delta,
        hook_mean_short_term=hook_mean_short_term,
        baseline_mean_short_term=baseline_mean_short_term,
    )


def compute_tempo_metrics(audio: np.ndarray, sr: int) -> TempoMetrics:
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    tempo_bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    if tempogram.size == 0:
        tempo_confidence = 0.0
    else:
        tempo_confidence = float(tempogram.max() / np.sum(tempogram))

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    hook_onsets = onset_times[onset_times < 5]
    onset_density_hook = float(len(hook_onsets) / 5.0) if len(hook_onsets) else 0.0

    if beat_times.size > 2:
        intervals = np.diff(beat_times)
        tempo_stability = float(np.std(intervals))
    else:
        tempo_stability = None

    harmonic, percussive = librosa.effects.hpss(audio)
    percussive_energy = np.sum(np.abs(percussive))
    total_energy = np.sum(np.abs(percussive)) + np.sum(np.abs(harmonic))
    percussive_ratio_overall = float(percussive_energy / total_energy) if total_energy > 0 else 0.0

    hook_samples = int(min(5 * sr, len(audio)))
    hook_percussive_energy = np.sum(np.abs(percussive[:hook_samples]))
    hook_total_energy = hook_percussive_energy + np.sum(np.abs(harmonic[:hook_samples]))
    percussive_ratio_hook = float(hook_percussive_energy / hook_total_energy) if hook_total_energy > 0 else 0.0

    return TempoMetrics(
        tempo_bpm=float(tempo_bpm),
        tempo_confidence=tempo_confidence,
        onset_density_hook=onset_density_hook,
        percussive_ratio_overall=percussive_ratio_overall,
        percussive_ratio_hook=percussive_ratio_hook,
        tempo_stability=tempo_stability,
        beat_times=beat_times,
    )


def compute_dynamics_metrics(audio: np.ndarray, sr: int) -> DynamicsMetrics:
    hop_length = 512
    frame_rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms_dbfs = 20 * np.log10(np.maximum(frame_rms, 1e-12))
    rms_times = librosa.frames_to_time(np.arange(len(frame_rms)), sr=sr, hop_length=hop_length)

    def _transient_punch(window: Tuple[float, float]) -> float:
        mask = (rms_times >= window[0]) & (rms_times < window[1])
        if not mask.any():
            return float("nan")
        window_rms = rms_dbfs[mask]
        p95 = np.nanpercentile(window_rms, 95)
        p50 = np.nanpercentile(window_rms, 50)
        return float(p95 - p50)

    transient_punch_hook = _transient_punch((0, 5))
    transient_punch_body = _transient_punch((5, 15))

    return DynamicsMetrics(
        rms_dbfs=rms_dbfs,
        rms_times=rms_times,
        transient_punch_hook=transient_punch_hook,
        transient_punch_body=transient_punch_body,
    )


def format_metric(value: float, precision: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "нет данных"
    return f"{value:.{precision}f}{suffix}"


def build_waveform_figure(audio: np.ndarray, sr: int) -> go.Figure:
    times = np.arange(len(audio)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=audio, mode="lines", name="Амплитуда"))
    fig.update_layout(
        title="Временная форма волны (амплитуда во времени)",
        xaxis_title="Время, с",
        yaxis_title="Амплитуда (норм. к 0 dBFS)",
        template="plotly_white",
    )
    return fig


def build_loudness_figure(metrics: LoudnessMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics.momentary_times,
            y=metrics.momentary_lufs,
            mode="lines",
            name="Momentary (400 мс)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.short_term_times,
            y=metrics.short_term_lufs,
            mode="lines",
            name="Short-Term (3 с)",
        )
    )
    fig.update_layout(
        title="Громкость во времени (LUFS)",
        xaxis_title="Время, с",
        yaxis_title="LUFS (дБ относительно полношкального сигнала)",
        template="plotly_white",
        legend_title="Окно расчёта",
    )
    return fig


def build_tempogram_figure(audio: np.ndarray, sr: int) -> go.Figure:
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)

    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=freqs,
            z=tempogram,
            colorscale="Viridis",
            colorbar=dict(title="Энергия"),
        )
    )
    fig.update_layout(
        title="Темпограмма (оценка темпа во времени)",
        xaxis_title="Время, с",
        yaxis_title="Темп, BPM",
        template="plotly_white",
    )
    return fig


def build_rms_figure(dynamics: DynamicsMetrics) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(x=dynamics.rms_times, y=dynamics.rms_dbfs, mode="lines", name="RMS")
    )
    fig.update_layout(
        title="Огибающая RMS (энергия сигнала во времени)",
        xaxis_title="Время, с",
        yaxis_title="RMS, dBFS",
        template="plotly_white",
    )
    return fig


def render_html(
    loudness: LoudnessMetrics,
    tempo: TempoMetrics,
    dynamics: DynamicsMetrics,
    waveform_fig: go.Figure,
    loudness_fig: go.Figure,
    tempogram_fig: go.Figure,
    rms_fig: go.Figure,
    video_path: Path,
) -> str:
    plotly_js = "<script src=\"https://cdn.plot.ly/plotly-2.29.1.min.js\"></script>"
    sections: List[str] = [
        "<h1>Аналитический отчёт по аудиодорожке</h1>",
        f"<p><strong>Источник видео:</strong> {video_path}</p>",
        "<p>Отчёт описывает громкость, темп и динамические контрасты аудио. Формулировки даны простым языком, чтобы даже без опыта в аудио было понятно, что означают показатели и как они связаны с восприятием.</p>",
        "<h2>Быстрый обзор метрик</h2>",
        "<ul>"
        f"<li><strong>Integrated Loudness (LUFS-I)</strong>: {format_metric(loudness.integrated_lufs, suffix=' LUFS')} — средняя воспринимаемая громкость всего ролика по стандарту ITU-R BS.1770 с K-взвешиванием.</li>"
        f"<li><strong>Loudness Range (LRA)</strong>: {format_metric(loudness.lra, suffix=' LU')} — разброс короткосрочной громкости (между 10-м и 95-м перцентилями после гейтинга), отражает макро-контраст.</li>"
        f"<li><strong>True Peak</strong>: {format_metric(loudness.true_peak_dbfs, suffix=' dBFS')} — максимальный пик после апсемплинга, важен для контроля клиппинга.</li>"
        f"<li><strong>Crest Factor</strong>: {format_metric(loudness.crest_factor_db, suffix=' dB')} — разница между пиком и среднеквадратичным уровнем; показывает «ударность».</li>"
        f"<li><strong>Hook Loudness Δ</strong>: {format_metric(loudness.hook_loudness_delta, suffix=' LU')} — насколько средняя громкость первых 5 секунд выше (или ниже) фона 5–15 с.</li>"
        f"<li><strong>Темп</strong>: {format_metric(tempo.tempo_bpm, suffix=' BPM')} с уверенностью {format_metric(tempo.tempo_confidence * 100, precision=1, suffix='%')} — оценено по автокорреляции огибающей атак (onset envelope).</li>"
        f"<li><strong>Percussive Ratio</strong>: {format_metric(tempo.percussive_ratio_overall * 100, precision=1, suffix='%')} (всего) / {format_metric(tempo.percussive_ratio_hook * 100, precision=1, suffix='%')} (хук) — доля энергии перкуссивной части после HPSS-разделения.</li>"
        f"<li><strong>Onset Density</strong>: {format_metric(tempo.onset_density_hook, precision=2, suffix=' онсета/с')} в первых 5 с — частота «ударов»/атак, влияет на ощущение драйва.</li>"
        f"<li><strong>Tempo Stability</strong>: {format_metric(tempo.tempo_stability, precision=3, suffix=' с')} — стандартное отклонение интервалов между ударами; чем меньше, тем ровнее пульс.</li>"
        f"<li><strong>Transient Punch</strong>: {format_metric(dynamics.transient_punch_hook, suffix=' dB')} (0–5 с) / {format_metric(dynamics.transient_punch_body, suffix=' dB')} (5–15 с) — на сколько пиковые RMS-значения превышают типичные.</li>"
        "</ul>",
        "<h2>Что означают эти показатели</h2>",
        "<p><strong>LUFS</strong> — шкала, имитирующая человеческое восприятие громкости. Значения отрицательные: 0 LUFS соответствует максимально возможному уровню без искажений (0 dBFS). Integrated LUFS — усреднение по всему ролику; Short-Term (3 с) и Momentary (400 мс) показывают локальные изменения. LRA измеряет, насколько эти локальные значения разбросаны: низкий LRA — плоская динамика, высокий — есть контрасты.</p>",
        "<p><strong>True Peak</strong> ищет пики после передискретизации: сигнал может клипповать выше 0 dBFS даже если отсчёты ниже. <strong>Crest Factor</strong> = Peak − RMS. Большой crest означает яркие удары поверх среднего уровня; слишком низкий crest фактор — признак сильной компрессии.</p>",
        "<p><strong>Темп (BPM)</strong> оценивается по периодичности атак (onsets). <strong>Уверенность темпа</strong> — доля энергии главного пика темпограммы: если она низкая, пульс неустойчив или речь доминирует. <strong>Onset density</strong> — сколько атак в секунду, особенно в начале ролика; быстрый темп и частые атаки повышают возбуждение (arousal).</p>",
        "<p><strong>Percussive Ratio</strong> после HPSS показывает, сколько энергии несут ударные по сравнению с гармоникой. В хуке рост этой доли часто добавляет «драйва». <strong>Tempo Stability</strong> (std интервалов между ударами) отражает ровность грува: меньше — стабильнее.</p>",
        "<p><strong>Transient Punch</strong> сравнивает 95-й и 50-й процентили RMS: если разница большая, пики явно выделяются над средней энергией. Мы считаем отдельно для хука (0–5 с) и тела ролика (5–15 с), чтобы увидеть, есть ли дополнительный акцент в начале.</p>",
        "<h2>Визуализации</h2>",
        waveform_fig.to_html(include_plotlyjs=False, full_html=False, div_id="waveform"),
        loudness_fig.to_html(include_plotlyjs=False, full_html=False, div_id="loudness"),
        tempogram_fig.to_html(include_plotlyjs=False, full_html=False, div_id="tempogram"),
        rms_fig.to_html(include_plotlyjs=False, full_html=False, div_id="rms"),
        "<h2>Практические выводы</h2>",
        "<ol>"
        "<li>Стриминги нормализуют интегральную громкость примерно до −14 LUFS. Поэтому ключевое — форма огибающей и микро-динамика: LRA, Crest, Hook Loudness Δ и Transient Punch.</li>"
        "<li>Темп и перкуссивность работают только когда есть музыкальный материал. При низкой уверенности темпа (<50%) выводы делать осторожно.</li>"
        "<li>Hook Loudness Δ > 0 и заметный Transient Punch в первых 5 с помогают привлечь внимание без необходимости «перекачивать» весь трек.</li>"
        "<li>Оптимальный LRA контекст-зависим: слишком низкий утомляет, слишком высокий может отвлекать. Используйте показатель как гипотезу и калибруйте на своих метриках удержания.</li>"
        "</ol>",
    ]

    return "\n".join([plotly_js] + sections)


def generate_report(video_path: Path, output_html: Path) -> None:
    audio, sr = extract_audio(video_path)
    loudness = compute_loudness_metrics(audio, sr)
    tempo = compute_tempo_metrics(audio, sr)
    dynamics = compute_dynamics_metrics(audio, sr)

    waveform_fig = build_waveform_figure(audio, sr)
    loudness_fig = build_loudness_figure(loudness)
    tempogram_fig = build_tempogram_figure(audio, sr)
    rms_fig = build_rms_figure(dynamics)

    html = render_html(
        loudness=loudness,
        tempo=tempo,
        dynamics=dynamics,
        waveform_fig=waveform_fig,
        loudness_fig=loudness_fig,
        tempogram_fig=tempogram_fig,
        rms_fig=rms_fig,
        video_path=video_path,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"Отчёт сохранён в {output_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построить HTML-отчёт по аудио метрикам видео")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("data/videos/000-youtube.mp4"),
        help="Путь к видеофайлу",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/audio_report.html"),
        help="Путь для сохранения HTML отчёта",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(args.video_path, args.output_html)


if __name__ == "__main__":
    main()
