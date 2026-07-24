import { TextAttributes } from "@opentui/core";
import type { TopTok } from "../api";
import { probColors, theme } from "../theme";
import { clip, fit, pad, windowRange } from "../util";

const LABEL_W = 9; // "layer 10" plus a space
const CELL_W = 9;

/**
 * The lens grid: rows are layers with the deepest at the top, columns are input
 * positions, and each cell is that layer's top-1 guess for the token that comes
 * *after* that column, shaded by confidence. Watching a column go from noise at
 * the bottom to the answer at the top is the whole logit lens in one picture.
 *
 * Both axes are windowed rather than scrolled: the cursor is always on screen,
 * and `‹ ›` / `▲ ▼` mark content clipped off each edge.
 */
export function LensGrid(props: {
  layers: string[];
  tokens: string[];
  grid: TopTok[][][];
  /** Current layer (0 = embed). */
  rung: number;
  /** Selected input position. */
  pos: number;
  width: number;
  height: number;
}) {
  const { layers, tokens, grid, rung, pos, width, height } = props;

  // Deepest layer on top — same ordering as the old Textual grid.
  const order = layers.map((_, i) => i).reverse();
  const bodyH = Math.max(1, height - 2); // header + clipped-rows marker
  const rowWin = windowRange(order.length, order.indexOf(rung), bodyH);

  // Leave a column for the "clipped right" marker: a row wider than the panel
  // would wrap and overdraw the row above it.
  const maxCols = Math.max(1, Math.floor((width - LABEL_W - 1) / CELL_W));
  const colWin = windowRange(tokens.length, pos, maxCols);
  const cols: number[] = [];
  for (let p = colWin.start; p < colWin.end; p++) cols.push(p);

  const clippedLeft = colWin.start > 0;
  const clippedRight = colWin.end < tokens.length;
  const clippedAbove = rowWin.start > 0;
  const clippedBelow = rowWin.end < order.length;

  return (
    <box style={{ flexDirection: "column", width, height }}>
      <text>
        <span fg={theme.faint}>{pad(clippedLeft ? " ‹" : "", LABEL_W)}</span>
        {cols.map((p) => (
          <span
            key={p}
            fg={p === pos ? theme.text : theme.muted}
            attributes={p === pos ? TextAttributes.BOLD : 0}
          >
            {fit(tokens[p]!, CELL_W)}
          </span>
        ))}
        <span fg={theme.faint}>{clippedRight ? "›" : ""}</span>
      </text>

      {order.slice(rowWin.start, rowWin.end).map((r) => (
        <text key={r}>
          <span
            fg={r === rung ? theme.model : theme.faint}
            attributes={r === rung ? TextAttributes.BOLD : 0}
          >
            {pad(layers[r]!, LABEL_W)}
          </span>
          {cols.map((p) => {
            const top = grid[r]?.[p]?.[0];
            const { bg, fg } = probColors(top?.p ?? 0);
            const selected = r === rung && p === pos;
            return (
              <span
                key={p}
                fg={fg}
                bg={bg}
                attributes={selected ? TextAttributes.BOLD | TextAttributes.UNDERLINE : 0}
              >
                {fit(top?.t ?? "", CELL_W)}
              </span>
            );
          })}
        </text>
      ))}

      <text>
        <span fg={theme.faint}>
          {clip(
            `${clippedAbove ? " ▲" : "  "}${clippedBelow ? "▼" : " "}  each cell: that layer's guess for the token after that column`,
            width,
          )}
        </span>
      </text>
    </box>
  );
}
