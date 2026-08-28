import init, { analyze_observations, schema_version } from "./pkg/rhythm_map.js";

const elements = {
  analyze: document.querySelector("#analyze"),
  beatCount: document.querySelector("#beat-count"),
  changeCount: document.querySelector("#change-count"),
  globalBpm: document.querySelector("#global-bpm"),
  input: document.querySelector("#observations"),
  reset: document.querySelector("#reset"),
  result: document.querySelector("#result"),
  schema: document.querySelector("#wasm-schema"),
  segmentCount: document.querySelector("#segment-count"),
  segments: document.querySelector("#segments"),
  status: document.querySelector("#status"),
  tempoChart: document.querySelector("#tempo-chart"),
};

let fixtureText = "";

function setStatus(message, state) {
  elements.status.textContent = message;
  elements.status.dataset.state = state;
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function renderSegments(segments) {
  elements.segments.replaceChildren();
  for (const segment of segments) {
    const row = document.createElement("tr");
    const bpm = segment.kind === "constant"
      ? formatNumber(segment.start_bpm)
      : `${formatNumber(segment.start_bpm)} → ${formatNumber(segment.end_bpm)}`;
    for (const value of [
      `${formatNumber(segment.start_s, 2)}s`,
      `${formatNumber(segment.end_s, 2)}s`,
      segment.kind,
      bpm,
      formatNumber(segment.confidence, 2),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.segments.append(row);
  }
}

function renderTempoCurve(points) {
  const svg = elements.tempoChart;
  svg.replaceChildren();
  if (points.length < 2) {
    return;
  }
  const width = 720;
  const height = 220;
  const margin = 34;
  const minTime = points[0].time_s;
  const maxTime = points.at(-1).time_s;
  const bpms = points.map((point) => point.bpm);
  const minBpm = Math.floor(Math.min(...bpms) / 10) * 10 - 5;
  const maxBpm = Math.ceil(Math.max(...bpms) / 10) * 10 + 5;
  const x = (time) => margin + ((time - minTime) / (maxTime - minTime)) * (width - margin * 2);
  const y = (bpm) => height - margin - ((bpm - minBpm) / (maxBpm - minBpm)) * (height - margin * 2);
  const namespace = "http://www.w3.org/2000/svg";

  for (const bpm of [minBpm, (minBpm + maxBpm) / 2, maxBpm]) {
    const line = document.createElementNS(namespace, "line");
    line.setAttribute("x1", margin);
    line.setAttribute("x2", width - margin);
    line.setAttribute("y1", y(bpm));
    line.setAttribute("y2", y(bpm));
    line.setAttribute("class", "grid-line");
    svg.append(line);

    const label = document.createElementNS(namespace, "text");
    label.setAttribute("x", 2);
    label.setAttribute("y", y(bpm) + 4);
    label.textContent = `${formatNumber(bpm, 0)}`;
    svg.append(label);
  }

  const polyline = document.createElementNS(namespace, "polyline");
  polyline.setAttribute("points", points.map((point) => `${x(point.time_s)},${y(point.bpm)}`).join(" "));
  polyline.setAttribute("class", "tempo-line");
  svg.append(polyline);
}

function renderAnalysis(analysis) {
  elements.globalBpm.textContent = formatNumber(analysis.global_bpm);
  elements.beatCount.textContent = analysis.beats.length;
  elements.segmentCount.textContent = analysis.tempo_segments.length;
  elements.changeCount.textContent = analysis.change_points.length;
  renderTempoCurve(analysis.tempo_curve);
  renderSegments(analysis.tempo_segments);
  elements.result.textContent = JSON.stringify(analysis, null, 2);
}

function clearAnalysis() {
  elements.globalBpm.textContent = "—";
  elements.beatCount.textContent = "—";
  elements.segmentCount.textContent = "—";
  elements.changeCount.textContent = "—";
  elements.tempoChart.replaceChildren();
  elements.segments.replaceChildren();
}

function runAnalysis() {
  try {
    const observations = JSON.parse(elements.input.value);
    const analysis = analyze_observations(observations);
    renderAnalysis(analysis);
    setStatus("Analysis complete", "ready");
  } catch (error) {
    clearAnalysis();
    elements.result.textContent = String(error);
    setStatus("Input or analysis error", "error");
  }
}

async function boot() {
  try {
    const [response] = await Promise.all([fetch("./observations.json"), init()]);
    if (!response.ok) {
      throw new Error(`fixture request failed: HTTP ${response.status}`);
    }
    fixtureText = await response.text();
    elements.input.value = fixtureText;
    elements.schema.textContent = `Schema ${schema_version()}`;
    elements.analyze.disabled = false;
    setStatus("WebAssembly ready", "ready");
    runAnalysis();
  } catch (error) {
    elements.result.textContent = String(error);
    setStatus("Failed to initialize", "error");
  }
}

elements.analyze.addEventListener("click", runAnalysis);
elements.reset.addEventListener("click", () => {
  elements.input.value = fixtureText;
  runAnalysis();
});

await boot();
