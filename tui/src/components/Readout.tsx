import type { LensResponse, TopTok } from "../api";
import { theme } from "../theme";
import { bar, clip, fit, pct } from "../util";

const TOK_W = 11;
const PCT_W = 7;

/**
 * The selected cell, spelled out: what each lens thinks comes next, and what
 * that costs in bits.
 *
 * The J-lens readout only appears at the prediction position — J·h is defined
 * for the last position's content, which is the one the model is actually
 * using to predict.
 */
export function Readout(props: {
  resp: LensResponse;
  rung: number;
  pos: number;
  width: number;
  height: number;
}) {
  const { resp, rung, pos, width, height } = props;
  const atPred = pos === resp.tokens.length - 1;
  const classic = resp.grid[rung]?.[pos] ?? [];
  const transported = resp.jtop && atPred ? resp.jtop[rung] : null;

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
    const label = `−log₂ p(${resp.pred.token}) = ${(bits ?? 0).toFixed(2)}b `;
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
      {atPred ? bitsLine(resp.bits[rung], theme.data) : null}

      {transported ? (
        <>
          <text>
            <span fg={theme.faint}>{clip("J-LENS — transported through the rest first", width)}</span>
          </text>
          {rows(transported, theme.model)}
          {bitsLine(resp.jbits?.[rung], theme.model)}
        </>
      ) : resp.jtop ? (
        <text>
          <span fg={theme.faint}>{clip("J-lens reads at the prediction position — press → to it", width)}</span>
        </text>
      ) : null}

      <text>
        <span fg={theme.faint}>
          {clip(
            `answer: ${resp.pred.token} (${(resp.pred.p * 100).toFixed(1)}%, ${resp.pred.bits.toFixed(2)}b) · uniform ${resp.uniform.toFixed(1)}b`,
            width,
          )}
        </span>
      </text>
    </box>
  );
}
