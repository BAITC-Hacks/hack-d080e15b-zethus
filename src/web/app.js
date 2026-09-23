"use strict";

const $ = (id) => document.getElementById(id);
const player = $("reply-audio");
const canRecord = Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
let busy = false;
let recorder = null;
let current = null;
let playback = null;
let measurements = [];
let traceTimer = null;

function status(text) { $("status").textContent = text; }
function setBusy(value) {
  busy = value;
  for (const id of ["text", "send", "reset"]) $(id).disabled = value;
  $("record").disabled = !canRecord || (value && recorder?.state !== "recording");
}
function message(text, kind = "assistant") {
  const item = document.createElement("article");
  item.className = `message ${kind}`;
  const label = document.createElement("span");
  label.textContent = kind === "user" ? "ВЫ" : "SAQTA";
  const body = document.createElement("p");
  body.textContent = text;
  item.append(label, body);
  $("messages").append(item);
  $("messages").scrollTop = $("messages").scrollHeight;
}
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Сервис временно недоступен.");
  return result;
}
function ms(value) { return typeof value === "number" ? `${Math.round(value)} мс` : "—"; }
function render(trace) {
  $("turn-label").textContent = `Реплика ${trace.turn}`;
  $("scenarios").textContent = trace.scenarios.join(" · ") || "Без маршрутизации";
  $("language").textContent = trace.language === "kk" ? "Қазақша" : "Русский";
  const decisions = trace.routing?.scenarios || [];
  $("reason").textContent = decisions.length
    ? decisions.map((entry) => `${entry.scenario_id}: ${entry.reason} (${Math.round(entry.confidence * 100)}%)`).join("\n")
    : trace.reason;
  $("alternatives").textContent = trace.alternatives?.length
    ? trace.alternatives.map((entry) => `${entry.scenario_id} · ${Math.round(entry.confidence * 100)}% — ${entry.reason}`).join("\n")
    : ({ pending: "Объяснение и альтернативы готовятся…", unavailable: "Объяснение недоступно. Выбор сценария сохранён.",
      busy: "Объясняющая модель занята. Выбор сценария сохранён.", ready: "Близкие альтернативы не предложены." }[trace.explanation_status]
      || "Локальный шаг диалога: повторный выбор сценария не нужен.");
  $("router-time").textContent = ms(trace.latency_ms.router);
  $("playback-time").textContent = ms(trace.latency_ms.total);
  $("stages").replaceChildren();
  for (const [key, label] of [["stt", "Распознавание"], ["triage", "Понимание реплики"],
    ["router", "Выбор сценария"], ["response", "Формулировка ответа"], ["dialog", "Диалог целиком"],
    ["tts_first_chunk", "Первый фрагмент озвучки"], ["tts", "Весь синтез речи"],
    ["explanation", "Объяснение (в фоне)"]]) {
    const row = document.createElement("div");
    row.className = "stage";
    const name = document.createElement("span"); name.textContent = label;
    const value = document.createElement("b"); value.textContent = ms(trace.latency_ms[key]);
    row.append(name, value); $("stages").append(row);
  }
  $("actions").textContent = JSON.stringify({ slots: trace.slots, actions: trace.actions,
    queued: trace.queued, suspended: trace.suspended, awaiting_confirmation: trace.awaiting_confirmation }, null, 2);
  $("trace").textContent = JSON.stringify(trace, null, 2);
  $("download").disabled = false;
}
async function refreshTrace(id) {
  try {
    const trace = await api(`/api/trace/${id}`);
    if (current?.turn_id !== id) return;
    // Браузер и сервер имеют разные часы: переносим только измеренные интервалы.
    const client = current.trace.client_latency_ms;
    if (client) {
      trace.client_latency_ms = client;
      trace.latency_ms.total = client.speech_to_playback_estimate;
    }
    current.trace = trace;
    render(trace);
  } catch { /* Ответ и уже полученная трассировка остаются доступны. */ }
}
async function pollExplanation(id, remaining = 120) {
  if (current?.turn_id !== id || !remaining) return;
  await refreshTrace(id);
  if (current?.turn_id === id && current.trace.explanation_status === "pending") {
    traceTimer = setTimeout(() => pollExplanation(id, remaining - 1), 1000);
  }
}

player.addEventListener("playing", () => {
  if (!playback || playback.measured || current?.turn_id !== playback.id) return;
  playback.measured = true;
  const now = performance.now();
  const duration = playback.speechEnd === null ? null : Math.round(now - playback.speechEnd);
  current.trace.client_latency_ms = {
    speech_to_playback_estimate: duration,
    stop_to_playback: playback.stopped === null ? null : Math.round(now - playback.stopped),
    request_to_playback: Math.round(now - playback.started),
    excluded_from_median: playback.manual || duration === null,
    measurement: "Browser playing event; speech end estimated by microphone energy",
  };
  current.trace.latency_ms.total = duration;
  $("playback-note").textContent = duration === null ? "текстовый ввод: без оценки речи" : "конец речи ≈ → playing";
  if (duration !== null && !playback.manual) {
    measurements.push(duration);
    const sorted = [...measurements].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    $("median").textContent = `Медиана голосовых запросов: ${ms(median)} · n=${sorted.length}. Конец речи оценивается по микрофону; для жюри проверьте секундомером.`;
  }
  render(current.trace);
  refreshTrace(playback.id);
});
player.addEventListener("ended", () => { if (current) refreshTrace(current.turn_id); });
player.addEventListener("error", () => {
  if (player.getAttribute("src")) status("Не удалось воспроизвести голос. Ответ доступен текстом.");
});

async function submit(payload, timing = {}) {
  setBusy(true);
  player.pause();
  clearTimeout(traceTimer);
  playback = null;
  status(payload.audio ? "Распознаём речь и выбираем сценарий…" : "Выбираем сценарий…");
  const started = performance.now();
  try {
    const response = await api("/api/turn", payload);
    current = response;
    message(response.transcript, "user");
    message(response.text);
    $("playback-note").textContent = "ожидаем начало воспроизведения";
    render(response.trace);
    pollExplanation(response.turn_id);
    if ($("voice").checked) {
      playback = { id: response.turn_id, started, stopped: timing.stopped ?? null,
        speechEnd: timing.speechEnd ?? null, measured: false, manual: false };
      player.src = response.audio_url;
      try { await player.play(); status("Можно задать следующий вопрос."); }
      catch {
        if (playback) playback.manual = true;
        status("Нажмите ▶ на плеере. Ручной запуск не попадёт в медиану задержки.");
      }
    } else status("Ответ готов. Озвучивание выключено.");
  } catch (error) {
    message(error.message || "Соединение прервалось. Повторите запрос.", "error");
    status("Не удалось обработать реплику.");
  } finally { setBusy(false); }
}

$("text-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("text").value.trim();
  if (busy || !text) return;
  $("text").value = "";
  submit({ text });
});

$("record").addEventListener("click", async () => {
  if (recorder?.state === "recording") { recorder.stop(); return; }
  if (busy) return;
  setBusy(true);
  player.pause();
  let stream = null;
  let context = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    if (!mime) throw new Error("Браузер не поддерживает нужный формат записи. Попробуйте Chrome или Firefox.");
    recorder = new MediaRecorder(stream, { mimeType: mime });
    const chunks = [];
    const recordingStarted = performance.now();
    let lastSpeech = null;
    let frame = null;
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    context = new AudioContextClass();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    function listen() {
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
      if (rms > 0.015) lastSpeech = performance.now();
      frame = requestAnimationFrame(listen);
    }
    recorder.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
    const timer = setTimeout(() => { if (recorder?.state === "recording") recorder.stop(); }, 60000);
    let recordingError = false;
    recorder.addEventListener("error", () => {
      recordingError = true;
      status("Запись прервалась. Попробуйте ещё раз или напишите текст.");
    });
    recorder.addEventListener("stop", async () => {
      const stopped = performance.now();
      clearTimeout(timer);
      cancelAnimationFrame(frame);
      stream.getTracks().forEach((track) => track.stop());
      context.close();
      $("record").classList.remove("recording");
      $("record").textContent = "● Говорить";
      $("record").disabled = true;
      try {
        if (recordingError) throw new Error("Запись прервалась. Попробуйте ещё раз.");
        const blob = new Blob(chunks, { type: mime });
        if (!blob.size || blob.size > 10 * 1024 * 1024) throw new Error("Запись пустая или больше 10 МБ.");
        const audio = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result.split(",")[1]);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        await submit({ audio, mime, duration_ms: Math.round(stopped - recordingStarted) },
          { stopped, speechEnd: lastSpeech });
      } catch (error) { status(error.message || "Не удалось прочитать запись."); setBusy(false); }
    }, { once: true });
    recorder.start();
    listen();
    $("record").classList.add("recording");
    $("record").textContent = "■ Стоп";
    $("record").disabled = false;
    status("Слушаю. Нажмите «Стоп», когда закончите.");
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    context?.close();
    status(error.name === "NotAllowedError" ? "Разрешите доступ к микрофону или напишите текст." : error.message);
    setBusy(false);
  }
});

$("reset").addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  player.pause();
  try {
    await api("/api/reset", {});
    clearTimeout(traceTimer);
    current = null; playback = null; measurements = [];
    $("messages").replaceChildren();
    message("Новый разговор. Здравствуйте! Сәлеметсіз бе!");
    for (const id of ["scenarios", "router-time", "playback-time"]) $(id).textContent = "—";
    $("trace").textContent = "{}";
    $("reason").textContent = "Ожидаем новую реплику.";
    $("alternatives").textContent = "Пока нет данных.";
    $("actions").textContent = "Нет активной задачи.";
    $("stages").replaceChildren();
    $("turn-label").textContent = "Ожидаем реплику";
    $("language").textContent = "Русский / қазақша";
    $("playback-note").textContent = "замер в браузере";
    $("median").textContent = "Медиана сброшена. Начните новую серию голосовых запросов.";
    $("download").disabled = true;
    player.removeAttribute("src"); player.load();
    status("Готов к разговору.");
  } catch (error) { status(error.message); }
  finally { setBusy(false); }
});

$("download").addEventListener("click", () => {
  if (!current) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(current.trace, null, 2)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = `saqta-turn-${current.trace.turn}.json`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => { if (!busy) { $("text").value = button.dataset.example; $("text").focus(); } });
});

api("/api/session").then((data) => {
  $("demo-help").textContent = data.demo;
  $("snapshot").textContent = data.date;
  status(canRecord ? "Готов к разговору." : "Микрофон недоступен. Откройте localhost в современном браузере.");
  setBusy(false);
}).catch((error) => status(`Нет связи с сервером: ${error.message}`));
