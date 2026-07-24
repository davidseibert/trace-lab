import { cursorColumn, renderChart, type Series } from "../chart";
import { mix, theme } from "../theme";
import { clip } from "../util";

/**
 * Code length by depth: −log₂ p of the token the finished model actually
 * predicts, read off every rung of the residual stream. The curve falling is
 * the MDL statement of "the prediction sharpens with depth" — the code for the
 * next token shortens as the stream is refined. The dashed line is log₂V, what
 * a layer that has learned nothing about this context would pay.
 */
export function DepthChart(props: {
  bits: number[];
  jbits: number[] | null;
  uniform: number;
  rung: number;
  predToken: string;
  width: number;
  height: number;
}) {
  const { bits, jbits, uniform, rung, predToken, width, height } = props;

  const max = Math.max(1, uniform * 1.15, ...bits, ...(jbits ?? []));
  // Classic lens last so it keeps its colour wherever the two curves cross.
  const series: Series[] = [];
  if (jbits) series.push({ values: jbits, fg: theme.model });
  series.push({ values: bits, fg: theme.data });

  const plotH = Math.max(1, height - 2); // legend + axis row
  const rows = renderChart({
    width,
    height: plotH,
    series,
    max,
    reference: { value: uniform, fg: theme.faint },
    cursor: rung,
    cursorBg: mix(theme.bg, theme.border2, 0.7),
  });

  // The scale first, the token it's measuring only if there's room left —
  // truncating mid-token reads worse than not showing it at all.
  const room = width - (jbits ? 16 : 12);
  const scale = `  ┈ ${uniform.toFixed(1)}b  top ${max.toFixed(0)}b`;
  const named = `${scale}  “${predToken}”`;
  const legend = clip(named.length <= room ? named : scale, room);

  // The axis rule is drawn to exactly `width`: a line even one character wider
  // than its box wraps, steals a row, and overdraws the panel above it.
  const cursorX = cursorColumn(rung, [{ values: bits, fg: theme.data }], width);
  const before = "─".repeat(Math.max(0, cursorX));
  const after = "─".repeat(Math.max(0, width - cursorX - 1));

  return (
    <box style={{ flexDirection: "column", width, height }}>
      <text>
        <span fg={theme.data}>━ lens</span>
        {jbits ? <span fg={theme.model}>{"  ━ J-lens"}</span> : <span fg={theme.faint}>{"  no J"}</span>}
        <span fg={theme.faint}>{legend}</span>
      </text>

      {rows.map((row, i) => (
        <text key={i}>
          {row.map((cell, j) => (
            <span key={j} fg={cell.fg} bg={cell.bg}>
              {cell.char}
            </span>
          ))}
        </text>
      ))}

      <text>
        <span fg={theme.border2}>{before}</span>
        <span fg={theme.model}>▲</span>
        <span fg={theme.border2}>{after}</span>
      </text>
    </box>
  );
}
