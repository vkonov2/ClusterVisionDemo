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
    beat_intervals: np.ndarray


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


def compute_true_peak_dbfs(audio: np.ndarray, sr: int) -> float:
    """Approximate true peak using 4x oversampling to catch inter-sample peaks."""
    if audio.size == 0:
        return float("nan")

    oversample_factor = 4
    try:
        upsampled = librosa.resample(audio, orig_sr=sr, target_sr=sr * oversample_factor)
    except Exception:
        upsampled = audio

    peak_linear = float(np.max(np.abs(upsampled)))
    if peak_linear <= 0:
        return float("nan")
    return float(20 * np.log10(peak_linear))


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
    true_peak_dbfs = compute_true_peak_dbfs(audio, sr)

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
        intervals = np.array([])
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
        beat_intervals=intervals,
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
    fig.add_hrect(
        y0=metrics.integrated_lufs - metrics.lra / 2 if np.isfinite(metrics.lra) else None,
        y1=metrics.integrated_lufs + metrics.lra / 2 if np.isfinite(metrics.lra) else None,
        fillcolor="LightSkyBlue",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="Диапазон LRA",
    )
    fig.add_hline(
        y=metrics.integrated_lufs,
        line=dict(color="black", dash="dash"),
        annotation_text="Integrated LUFS",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Громкость во времени (LUFS)",
        xaxis_title="Время, с",
        yaxis_title="LUFS (дБ относительно полношкального сигнала)",
        template="plotly_white",
        legend_title="Окно расчёта",
    )
    return fig


def build_lra_distribution_figure(metrics: LoudnessMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=metrics.short_term_lufs,
            name="Short-Term LUFS",
            nbinsx=40,
            marker_color="#1f77b4",
        )
    )
    if np.isfinite(metrics.lra):
        fig.add_vrect(
            x0=np.nanmedian(metrics.short_term_lufs) - metrics.lra / 2,
            x1=np.nanmedian(metrics.short_term_lufs) + metrics.lra / 2,
            fillcolor="LightSkyBlue",
            opacity=0.25,
            line_width=0,
            annotation_text="Диапазон LRA",
        )
    fig.add_vline(
        x=metrics.integrated_lufs,
        line=dict(color="black", dash="dash"),
        annotation_text="Integrated",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Распределение короткосрочной громкости (для расчёта LRA)",
        xaxis_title="LUFS",
        yaxis_title="Количество окон",
        template="plotly_white",
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


def build_beat_interval_figure(tempo: TempoMetrics) -> go.Figure:
    fig = go.Figure()
    if tempo.beat_times.size:
        fig.add_trace(
            go.Scatter(
                x=tempo.beat_times,
                y=np.pad(tempo.beat_intervals, (1, 0), constant_values=np.nan),
                mode="lines+markers",
                name="Интервалы между ударами",
            )
        )
    fig.update_layout(
        title="Стабильность темпа: интервалы между ударами",
        xaxis_title="Время удара, с",
        yaxis_title="Длительность интервала, с",
        template="plotly_white",
    )
    return fig


def build_percussive_ratio_figure(tempo: TempoMetrics) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Весь ролик", "Хук 0–5 с"],
                y=[
                    tempo.percussive_ratio_overall * 100,
                    tempo.percussive_ratio_hook * 100,
                ],
                marker_color=["#636efa", "#ef553b"],
                texttemplate="%{y:.1f}%",
            )
        ]
    )
    fig.update_layout(
        title="Доля перкуссивной энергии (HPSS)",
        yaxis_title="Перкуссивная энергия, %",
        template="plotly_white",
    )
    return fig


def build_rms_figure(dynamics: DynamicsMetrics) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(x=dynamics.rms_times, y=dynamics.rms_dbfs, mode="lines", name="RMS")
    )
    fig.add_vrect(x0=0, x1=5, fillcolor="LightGreen", opacity=0.15, line_width=0, annotation_text="Хук 0–5 с")
    fig.add_vrect(x0=5, x1=15, fillcolor="LightBlue", opacity=0.1, line_width=0, annotation_text="Фон 5–15 с")
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
    lra_fig: go.Figure,
    tempogram_fig: go.Figure,
    beat_interval_fig: go.Figure,
    percussive_ratio_fig: go.Figure,
    rms_fig: go.Figure,
    video_path: Path,
) -> str:
    plotly_js = "<script src=\"https://cdn.plot.ly/plotly-2.29.1.min.js\"></script>"
    body_sections: List[str] = [
        "<h1>Аналитический отчёт по аудиодорожке</h1>",
        f"<p><strong>Источник видео:</strong> {video_path}</p>",
        "<p>Отчёт собирает громкость, темп и динамические контрасты аудио. Все термины объяснены простым языком, чтобы читатель без опыта в звукотехнике понимал, что измерено и зачем.</p>",
        "<h2>Краткая сводка чисел</h2>",
        "<ul class=\"metric-list\">",
        f"<li><strong>Integrated Loudness (LUFS-I)</strong>: {format_metric(loudness.integrated_lufs, suffix=' LUFS')} — средняя воспринимаемая громкость всего ролика по стандарту ITU-R BS.1770 с K-взвешиванием.</li>",
        f"<li><strong>Loudness Range (LRA)</strong>: {format_metric(loudness.lra, suffix=' LU')} — разброс короткосрочной громкости (перцентиль 95 минус перцентиль 10 после гейтинга, то есть отсечения очень тихих фрагментов), показывает макро-контраст.</li>",
        f"<li><strong>True Peak</strong>: {format_metric(loudness.true_peak_dbfs, suffix=' dBFS')} — максимум сигнала после апсемплинга, нужен, чтобы поймать междискретные пики и избежать клиппинга.</li>",
        f"<li><strong>Crest Factor</strong>: {format_metric(loudness.crest_factor_db, suffix=' dB')} — разница между пиком и среднеквадратичным уровнем (RMS); характеризует «ударность».</li>",
        f"<li><strong>Hook Loudness Δ</strong>: {format_metric(loudness.hook_loudness_delta, suffix=' LU')} — изменение громкости первых 5 с относительно участка 5–15 с (база).</li>",
        f"<li><strong>Темп</strong>: {format_metric(tempo.tempo_bpm, suffix=' BPM')} с уверенностью {format_metric(tempo.tempo_confidence * 100, precision=1, suffix='%')} — найден по автокорреляции огибающей атак (onset envelope).</li>",
        f"<li><strong>Percussive Ratio</strong>: {format_metric(tempo.percussive_ratio_overall * 100, precision=1, suffix='%')}(весь ролик) / {format_metric(tempo.percussive_ratio_hook * 100, precision=1, suffix='%')} (хук) — доля энергии ударных после HPSS-разделения.</li>",
        f"<li><strong>Onset Density</strong>: {format_metric(tempo.onset_density_hook, precision=2, suffix=' онсета/с')} в 0–5 с — число атак в секунду, отражает насыщенность событий.</li>",
        f"<li><strong>Tempo Stability</strong>: {format_metric(tempo.tempo_stability, precision=3, suffix=' с')} — стандартное отклонение интервалов между ударами (чем меньше, тем ровнее ритм).</li>",
        f"<li><strong>Transient Punch</strong>: {format_metric(dynamics.transient_punch_hook, suffix=' dB')} (0–5 с) / {format_metric(dynamics.transient_punch_body, suffix=' dB')} (5–15 с) — насколько пики RMS выше типичных значений.</li>",
        "</ul>",
        "<h2>Глоссарий простыми словами</h2>",
        "<ul>",
        "<li><strong>ITU-R BS.1770, K-взвешивание</strong> — международный способ считать воспринимаемую громкость: сначала фильтр, имитирующий чувствительность уха (K-weighting), затем энергия в окнах 400 мс суммируется по каналам.</li>",
        "<li><strong>Гейтинг</strong> — отбрасываем слишком тихие фрагменты, чтобы они не занижали оценку. После этого считаем перцентили (10-й и 95-й) для LRA.</li>",
        "<li><strong>Апсемплинг</strong> — временное повышение частоты дискретизации, чтобы заметить пики между исходными отсчётами. Так оценивается True Peak.</li>",
        "<li><strong>Клиппинг</strong> — ситуация, когда сигнал выходит за 0 dBFS и искажается. True Peak помогает вовремя увидеть риск.</li>",
        "<li><strong>Автокорреляция огибающей атак</strong> — поиск повторяющегося ритма по энергии ударов. Так оцениваем темп (BPM).</li>",
        "<li><strong>HPSS</strong> — разделение сигнала на гармоническую (ноты, речь) и перкуссивную (удары) части; по перкуссивной считаем долю ударной энергии.</li>",
        "</ul>",
        "<h2>Почему уверенность темпа может быть 0%</h2>",
        "<p>Уверенность = доля энергии главного пика темпограммы. Если её почти нет, значит сигнал неритмичный (например, речь, шум или очень короткий отрезок), и модель не видит устойчивого пульса. В таких случаях выводы по темпу лучше не использовать.</p>",
        "<h2>Визуализации и как их читать</h2>",
        "<h3>Форма волны</h3>",
        "<p>Показывает амплитуду во времени — быстро видно тишину, всплески, монтажные стыки.</p>",
        waveform_fig.to_html(include_plotlyjs=False, full_html=False, div_id="waveform"),
        "<h3>Громкость во времени (Momentary/Short-Term) с Integrated и LRA</h3>",
        "<p>График объединяет мгновенную (400 мс) и короткосрочную (3 с) громкость. Пунктир — Integrated LUFS, голубая полоса — диапазон LRA. Если линия почти не выходит за полосу, динамика плоская; если гуляет, есть контрасты.</p>",
        loudness_fig.to_html(include_plotlyjs=False, full_html=False, div_id="loudness"),
        "<h3>Распределение Short-Term LUFS (как набирается LRA)</h3>",
        "<p>Гистограмма короткосрочной громкости с полосой LRA. Можно увидеть, есть ли много тихих/громких окон и насколько широко распределение.</p>",
        lra_fig.to_html(include_plotlyjs=False, full_html=False, div_id="lra"),
        "<h3>Темпограмма</h3>",
        "<p>Тепловая карта возможных темпов во времени. Яркий вертикальный гребень = устойчивый темп. Если карта равномерная, уверенность низкая.</p>",
        tempogram_fig.to_html(include_plotlyjs=False, full_html=False, div_id="tempogram"),
        "<h3>Стабильность темпа (интервалы между ударами)</h3>",
        "<p>Линия показывает длительность промежутков между ударами. Чем ровнее линия, тем стабильнее пульс. Разброс интервалов увеличивает Tempo Stability.</p>",
        beat_interval_fig.to_html(include_plotlyjs=False, full_html=False, div_id="beat-intervals"),
        "<h3>Доля перкуссивной энергии (HPSS)</h3>",
        "<p>Столбцы показывают, сколько энергии приходится на ударные во всём ролике и в хуке 0–5 с. Рост доли в начале обычно добавляет драйва.</p>",
        percussive_ratio_fig.to_html(include_plotlyjs=False, full_html=False, div_id="percussive"),
        "<h3>Огибающая RMS и Transient Punch</h3>",
        "<p>RMS показывает среднюю энергию в окнах. Полосы выделяют хук (0–5 с) и базу (5–15 с) — по ним считаются Transient Punch и Hook Loudness Δ.</p>",
        rms_fig.to_html(include_plotlyjs=False, full_html=False, div_id="rms"),
        "<h2>Практические выводы по данным этого ролика</h2>",
        "<ol>",
        f"<li>Integrated LUFS = {format_metric(loudness.integrated_lufs, suffix=' LUFS')}. На платформах громкость приведут к ≈ −14 LUFS, поэтому ценны контрасты: LRA = {format_metric(loudness.lra, suffix=' LU')} и Crest = {format_metric(loudness.crest_factor_db, suffix=' dB')} показывают, есть ли «дыхание» и удары.</li>",
        f"<li>Hook Loudness Δ = {format_metric(loudness.hook_loudness_delta, suffix=' LU')}. Если показатель около нуля или ниже, можно усилить вовлечённость, слегка подняв громкость/динамику в первых секундах без увеличения интегрального уровня.</li>",
        f"<li>Темп {format_metric(tempo.tempo_bpm, suffix=' BPM')} с уверенностью {format_metric(tempo.tempo_confidence * 100, precision=1, suffix='%')}. При низкой уверенности ритм неустойчив (часто из-за речи); для музыкального акцента стоит добавить явные удары или перкуссию.</li>",
        f"<li>Percussive Ratio: {format_metric(tempo.percussive_ratio_hook * 100, precision=1, suffix='%')} в хуке. Если цель — драйв, можно усилить ударные или подчеркнуть атаки (увеличить Onset Density = {format_metric(tempo.onset_density_hook, precision=2, suffix=' онсета/с')}).</li>",
        f"<li>Transient Punch: {format_metric(dynamics.transient_punch_hook, suffix=' dB')} в 0–5 с и {format_metric(dynamics.transient_punch_body, suffix=' dB')} в 5–15 с. Если значения малы, добавьте чуть больше атаки (эквалайзер/транзиент-шейпер) в ключевых моментах.</li>",
        "</ol>",
    ]

    body_html = "\n".join(body_sections)

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="ru">',
            "<head>",
            "<meta charset=\"UTF-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>Аналитический отчёт по аудиодорожке</title>",
            "<style>body{font-family:Arial,sans-serif;line-height:1.6;margin:24px;} h1,h2{color:#222;} ul,ol{margin-left:20px;} figure{margin:0 0 24px 0;} iframe{width:100%;} .metric-list li{margin-bottom:6px;}</style>",
            plotly_js,
            "</head>",
            "<body>",
            body_html,
            "</body>",
            "</html>",
        ]
    )

def generate_report(video_path: Path, output_html: Path) -> None:
    audio, sr = extract_audio(video_path)
    loudness = compute_loudness_metrics(audio, sr)
    tempo = compute_tempo_metrics(audio, sr)
    dynamics = compute_dynamics_metrics(audio, sr)

    waveform_fig = build_waveform_figure(audio, sr)
    loudness_fig = build_loudness_figure(loudness)
    lra_fig = build_lra_distribution_figure(loudness)
    tempogram_fig = build_tempogram_figure(audio, sr)
    beat_interval_fig = build_beat_interval_figure(tempo)
    percussive_ratio_fig = build_percussive_ratio_figure(tempo)
    rms_fig = build_rms_figure(dynamics)

    html = render_html(
        loudness=loudness,
        tempo=tempo,
        dynamics=dynamics,
        waveform_fig=waveform_fig,
        loudness_fig=loudness_fig,
        lra_fig=lra_fig,
        tempogram_fig=tempogram_fig,
        beat_interval_fig=beat_interval_fig,
        percussive_ratio_fig=percussive_ratio_fig,
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
