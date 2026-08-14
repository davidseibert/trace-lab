import type { LensResponse, TopTok } from "../api";
import { theme } from "../theme";
import { bar, clip, fit, pct } from "../util";

const TOK_W = 11;
const PCT_W = 7;

/** The selected column's ladder, as the App resolves it: from /lens at the
 * last column, lazily fetched from /column elsewhere, null while in flight. */
export interface Ladder {
  pred: { token: string; p: number; bits: number };
  bits: number[];
  jbits: number[] | null;
  jtop: TopTok[][] | null;
}

/**
 * The selected cell, spelled out: what each lens thinks comes next, and what
 * that costs in bits — for whichever column is selected. The ladder arrives
 * lazily for non-final columns (one JVP per rung server-side), so it may be
 * null for a beat after the cursor moves.
 */
export function Readout(props: {
  resp: LensResponse;
  ladder: Ladder | null;
  rung: number;
  pos: number;
  width: number;
  height: number;
}) {
  const { resp, ladder, rung, pos, width, height } = props;
  const classic = resp.grid[rung]?.[pos] ?? [];
  const transported = ladder?.jtop ? ladder.jtop[rung] : null;

  // Chrome: the context line, one header per group, one bits line per group,
  // and the model's-answer line at the bottom.
  const chrome = transported ? 6 : 4;
  const perGroup = Math.max(1, Math.floor((height - chrome) / (transported ? 2 : 1)));
  const barW = Math.max(4, width - TOK_W - PCT_W);

  const rows = (toks: TopTok[], fg: string) =>
    toks.slice(0, perGroup).map((tk, i) => (
      <text key={`${fg}-${i}`}>
        <span fg={theme.text}>{fit(tk.t, TOK_W)}</span>
        <span fg={fg}>{bar(tk.p, barW)}</span>
        <span fg={theme.muted}>{pct(tk.p)}</span>
      </text>
    ));

  const bitsLine = (bits: number | undefined, fg: string) => {
    const label = `−log₂ p(${ladder?.pred.token ?? "?"}) = ${(bits ?? 0).toFixed(2)}b `;
    return (
      <text>
        <span fg={theme.faint}>{clip(label, width)}</span>
        <span fg={fg}>{bar((bits ?? 0) / (resp.uniform * 1.3), Math.max(0, width - label.length))}</span>
      </text>
    );
  };

  return (
    <box style={{ flexDirection: "column", width, height }}>
      <text>
        <span fg={theme.faint}>{clip(`${resp.layers[rung]} @ “${resp.tokens[pos]}”`, width)}</span>
      </text>

      <text>
        <span fg={theme.faint}>{clip("LOGIT LENS — the rung decoded as-is", width)}</span>
      </text>
      {rows(classic, theme.data)}
      {ladder ? bitsLine(ladder.bits[rung], theme.data) : null}

      {transported ? (
        <>
          <text>
            <span fg={theme.faint}>{clip("J-LENS — transported through the rest first", width)}</span>
          </text>
          {rows(transported, theme.model)}
          {bitsLine(ladder?.jbits?.[rung], theme.model)}
        </>
      ) : !ladder ? (
        <text>
          <span fg={theme.faint}>{clip("computing this column's ladder…", width)}</span>
        </text>
      ) : null}

      {ladder ? (
        <text>
          <span fg={theme.faint}>
            {clip(
              `after this: ${ladder.pred.token} (${(ladder.pred.p * 100).toFixed(1)}%, ${ladder.pred.bits.toFixed(2)}b) · uniform ${resp.uniform.toFixed(1)}b`,
              width,
            )}
          </span>
        </text>
      ) : null}
    </box>
  );
}
