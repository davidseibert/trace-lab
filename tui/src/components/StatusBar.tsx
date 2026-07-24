import { theme } from "../theme";
import { clip } from "../util";

const KEYS = "↑↓ rung  ←→ pos  ⏎ run  ^L prompt  m model  j J-lens  g/G embed/final  q quit";

/** Transport line: where the cursor is, what it costs, and the key legend. */
export function StatusBar(props: {
  layer: string | null;
  rung: number;
  rungs: number;
  bits: number | null;
  jbits: number | null;
  message: string;
  width: number;
}) {
  const { layer, rung, rungs, bits, jbits, message, width } = props;

  const where = layer === null ? "" : `◀ ${layer} ▶  ${rung + 1}/${rungs}  `;
  const cost = bits === null ? "" : `${bits.toFixed(2)}b`;
  const jcost = jbits === null ? "" : `  J ${jbits.toFixed(2)}b`;
  const note = message ? `  ${message.replace(/^!/, "")}` : "";
  const room = width - where.length - cost.length - jcost.length;

  return (
    <box style={{ flexDirection: "column", height: 2, paddingLeft: 1, paddingRight: 1 }}>
      <text>
        <span fg={theme.muted}>{clip(where, width)}</span>
        <span fg={theme.data}>{cost}</span>
        <span fg={theme.model}>{jcost}</span>
        <span fg={message.startsWith("!") ? theme.bad : theme.faint}>{clip(note, Math.max(0, room))}</span>
      </text>
      <text>
        <span fg={theme.faint}>{clip(KEYS, width)}</span>
      </text>
    </box>
  );
}
