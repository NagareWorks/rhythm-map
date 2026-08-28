import { readFile } from "node:fs/promises";

import init, { analyze_observations, schema_version } from "./pkg/rhythm_map.js";

const wasm = await readFile(new URL("./pkg/rhythm_map_bg.wasm", import.meta.url));
await init({ module_or_path: wasm });

const observations = JSON.parse(
  await readFile(new URL("./observations.json", import.meta.url), "utf8"),
);
const analysis = analyze_observations(observations);

if (analysis.schema_version !== schema_version()) {
  throw new Error("WASM result schema does not match the exported schema version");
}
if (analysis.beats.length !== observations.beats.length) {
  throw new Error("WASM result did not preserve the fixture beat sequence");
}

console.log(JSON.stringify({
  schema_version: analysis.schema_version,
  beats: analysis.beats.length,
  tempo_segments: analysis.tempo_segments.length,
  change_points: analysis.change_points.length,
  source: analysis.source,
}));
