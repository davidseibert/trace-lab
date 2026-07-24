import { theme } from "../theme";

/**
 * Prompt, model, J-lens toggle and engine status — the whole control surface.
 *
 * Everything but the prompt is `flexShrink: 0`: a shrunk `<text>` wraps, and a
 * wrapped line inside a 1-row bar draws straight through the border.
 */
export function TopBar(props: {
  model: string;
  prompt: string;
  jlens: boolean;
  device: string | null;
  loading: boolean;
  focused: boolean;
  width: number;
  onInput: (value: string) => void;
}) {
  const { model, prompt, jlens, device, loading, focused, width } = props;
  const roomy = width >= 100;

  return (
    <box
      style={{
        flexDirection: "row",
        height: 3,
        border: true,
        borderStyle: "rounded",
        borderColor: theme.border,
        paddingLeft: 1,
        paddingRight: 1,
        gap: 1,
        alignItems: "center",
      }}
    >
      {roomy ? (
        <text style={{ flexShrink: 0 }}>
          <span fg={theme.data}>−log₂ p</span>
          <span fg={theme.faint}> · depth</span>
        </text>
      ) : null}
      <text style={{ flexShrink: 0 }}>
        <span fg={theme.text}>{model}</span>
      </text>
      <text style={{ flexShrink: 0 }}>
        <span fg={theme.faint}>❯</span>
      </text>
      {/* Enter is handled by the app's key handler, not the input's own submit
          event, so that ⏎ re-runs even when the prompt is unchanged. */}
      <box style={{ flexGrow: 1, flexShrink: 1, height: 1 }}>
        <input
          value={prompt}
          placeholder="type a prompt…"
          focused={focused}
          onInput={props.onInput}
          backgroundColor={theme.bg}
          focusedBackgroundColor={theme.panel}
          textColor={theme.text}
          cursorColor={theme.data}
        />
      </box>
      <text style={{ flexShrink: 0 }}>
        <span fg={jlens ? theme.model : theme.faint}>{jlens ? "J ✓" : "J ✗"}</span>
      </text>
      <text style={{ flexShrink: 0 }}>
        <span fg={device === null ? theme.bad : loading ? theme.data : theme.ok}>
          {device === null ? "offline" : loading ? "running…" : roomy ? `engine · ${device}` : device}
        </span>
      </text>
    </box>
  );
}
