import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { useCallback, useEffect, useRef, useState } from "react";

import { ENGINE_URL, fetchHealth, fetchLens, type LensResponse } from "./api";
import { DepthChart } from "./components/DepthChart";
import { LensGrid } from "./components/LensGrid";
import { Readout } from "./components/Readout";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { theme } from "./theme";

const DEFAULT_PROMPT = "The Eiffel Tower is in the city of";

export function App() {
  const renderer = useRenderer();
  const { width: W, height: H } = useTerminalDimensions();

  const [device, setDevice] = useState<string | null>(null); // null = engine unreachable
  const [models, setModels] = useState<string[]>(["gpt2"]);
  const [model, setModel] = useState("gpt2");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [jlens, setJlens] = useState(true);
  const [rollout, setRollout] = useState(0);

  const [resp, setResp] = useState<LensResponse | null>(null);
  const [rung, setRung] = useState(0);
  const [pos, setPos] = useState(0);

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [focus, setFocus] = useState<"prompt" | "grid">("grid");

  // The prompt lives in the Input renderable while it's being typed; only a
  // submit or a run promotes it to the value we send.
  const draft = useRef(prompt);

  const checkHealth = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setDevice(h.device);
      setModels(h.models);
      setModel((m) => (h.models.includes(m) ? m : h.default));
      return true;
    } catch {
      setDevice(null);
      return false;
    }
  }, []);

  const run = useCallback(
    async (over: { model?: string; prompt?: string; jlens?: boolean; rollout?: number } = {}) => {
      const req = {
        model: over.model ?? model,
        prompt: (over.prompt ?? draft.current).trim(),
        jlens: over.jlens ?? jlens,
        rollout: over.rollout ?? rollout,
      };
      if (!req.prompt) return;
      setLoading(true);
      setMessage(
        `${req.model} · ${req.jlens ? "logit + J-lens" : "logit lens"}` +
          `${req.rollout ? ` · rollout ${req.rollout}` : ""} …`,
      );
      const started = Date.now();
      try {
        const r = await fetchLens(req);
        setResp(r);
        // Start at the top rung: the model's real answer, then scrub down to
        // watch it form.
        setRung(r.layers.length - 1);
        setPos(r.tokens.length - 1);
        setMessage(`${r.layers.length} rungs in ${((Date.now() - started) / 1000).toFixed(1)}s`);
        setDevice((d) => d ?? "up");
      } catch (e) {
        setMessage(`!${e instanceof Error ? e.message : String(e)}`);
        await checkHealth();
      } finally {
        setLoading(false);
      }
    },
    [model, jlens, rollout, checkHealth],
  );

  // On entry: if the engine answers, run the default prompt so the lens is
  // already alive when you look at it.
  useEffect(() => {
    void checkHealth().then((ok) => {
      if (ok) void run();
    });
  }, []);

  useKeyboard((key) => {
    if (key.name === "tab") {
      setFocus((f) => (f === "prompt" ? "grid" : "prompt"));
      return;
    }
    if (focus === "prompt") {
      if (key.name === "escape") setFocus("grid");
      if (key.name === "return") {
        setPrompt(draft.current);
        setFocus("grid");
        void run({ prompt: draft.current });
      }
      return; // every other key belongs to the input
    }

    switch (key.name) {
      case "q":
      case "escape":
        renderer.destroy();
        process.exit(0);
        return;
      case "l":
        if (key.ctrl) setFocus("prompt");
        return;
      case "return":
        void run();
        return;
      case "j": {
        const next = !jlens;
        setJlens(next);
        void run({ jlens: next });
        return;
      }
      case "r": {
        // Cycle the server-side rollout: how many tokens the engine greedily
        // decodes and appends before lensing, one grid column per token.
        const steps = [0, 3, 8, 16];
        const next = steps[(steps.indexOf(rollout) + 1) % steps.length]!;
        setRollout(next);
        void run({ rollout: next });
        return;
      }
      case "m": {
        const next = models[(models.indexOf(model) + 1) % models.length]!;
        setModel(next);
        void run({ model: next });
        return;
      }
      case "g":
        if (!resp) return;
        setRung(key.shift ? resp.layers.length - 1 : 0);
        return;
      case "up":
        if (resp) setRung((r) => Math.min(resp.layers.length - 1, r + 1));
        return;
      case "down":
        if (resp) setRung((r) => Math.max(0, r - 1));
        return;
      case "left":
        setPos((p) => Math.max(0, p - 1));
        return;
      case "right":
        if (resp) setPos((p) => Math.min(resp.tokens.length - 1, p + 1));
        return;
    }
  });

  // Panel geometry. Explicit sizes beat measuring renderables here: the two
  // content panels rasterise text themselves and need to know their box first.
  const sideW = Math.max(38, Math.min(58, Math.floor(W * 0.4)));
  const gridW = Math.max(24, W - sideW - 1);
  const bodyH = Math.max(8, H - 3 /* topbar */ - 2 /* status */);
  const chartH = Math.max(8, Math.min(14, Math.floor(bodyH * 0.45)));
  const readH = Math.max(6, bodyH - chartH);

  const panel = {
    border: true,
    borderStyle: "rounded" as const,
    borderColor: theme.border,
    paddingLeft: 1,
    paddingRight: 1,
  };

  return (
    <box style={{ flexDirection: "column", width: W, height: H, backgroundColor: theme.bg }}>
      <TopBar
        model={model}
        prompt={prompt}
        jlens={jlens}
        rollout={rollout}
        device={device}
        loading={loading}
        focused={focus === "prompt"}
        width={W}
        onInput={(v) => {
          draft.current = v;
        }}
      />

      {resp ? (
        <box style={{ flexDirection: "row", height: bodyH, gap: 1 }}>
          <box title="lens grid" style={{ ...panel, width: gridW, height: bodyH }}>
            <LensGrid
              layers={resp.layers}
              tokens={resp.tokens}
              grid={resp.grid}
              nPrompt={resp.n_prompt ?? resp.tokens.length}
              rung={rung}
              pos={pos}
              width={gridW - 4}
              height={bodyH - 2}
            />
          </box>

          <box style={{ flexDirection: "column", width: sideW, height: bodyH }}>
            <box title="code length by depth" style={{ ...panel, width: sideW, height: chartH }}>
              <DepthChart
                bits={resp.bits}
                jbits={resp.jbits}
                uniform={resp.uniform}
                rung={rung}
                predToken={resp.pred.token}
                width={sideW - 4}
                height={chartH - 2}
              />
            </box>
            <box title="rung readout" style={{ ...panel, width: sideW, height: readH }}>
              <Readout resp={resp} rung={rung} pos={pos} width={sideW - 4} height={readH - 2} />
            </box>
          </box>
        </box>
      ) : (
        <Offline device={device} loading={loading} message={message} height={bodyH} />
      )}

      <StatusBar
        layer={resp ? (resp.layers[rung] ?? null) : null}
        rung={rung}
        rungs={resp?.layers.length ?? 0}
        bits={resp?.bits[rung] ?? null}
        jbits={resp?.jbits?.[rung] ?? null}
        message={message}
        width={W - 2}
      />
    </box>
  );
}

function Offline(props: { device: string | null; loading: boolean; message: string; height: number }) {
  const { device, loading, message, height } = props;
  return (
    <box
      style={{
        height,
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        gap: 1,
      }}
    >
      {device === null ? (
        <>
          <text>
            <span fg={theme.bad}>engine offline</span>
          </text>
          <text>
            <span fg={theme.faint}>{`no lens engine at ${ENGINE_URL} — start it with:`}</span>
          </text>
          <text>
            <span fg={theme.text}>make engine</span>
          </text>
          <text>
            <span fg={theme.faint}>then press ⏎ to retry</span>
          </text>
        </>
      ) : (
        <text>
          <span fg={theme.faint}>{loading ? "running…" : message || "press ⏎ to run"}</span>
        </text>
      )}
    </box>
  );
}
