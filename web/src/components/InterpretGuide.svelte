<script lang="ts">
  /**
   * Field guide to the instruments: what each reading legitimately supports,
   * and the seductive misreadings. Anchored to real observations from this
   * repo's own sessions so the caveats stay memorable. Rendered inside a
   * default-collapsed panel on the real-model lenses.
   */
  interface Item {
    ok: boolean; // ✓ valid reading vs ✗ false path
    text: string;
  }
  interface Section {
    id: string;
    title: string;
    items: Item[];
  }

  const ALL: Section[] = [
    {
      id: 'bits',
      title: 'per-token code length (yellow shading, strip chart)',
      items: [
        { ok: true, text: 'Each token is priced −log₂p under the model’s TRUE distribution. Temperature changes which path is walked — never the prices.' },
        { ok: true, text: 'Spikes are forks: several continuations were live (a plan choice, a discourse pivot, formatting freedom). Valleys are execution: context already determined the token.' },
        { ok: false, text: 'A spike is not confusion or an error. “So” costing 1.9b means the sentence had options, not that the model struggled.' },
        { ok: false, text: 'At temp > 0, a bright token may be an unlucky draw, not a preference — “ violence” was emitted at p=0.34% while the model wanted “ protests” at 38%. Check p before reading intent.' },
        { ok: true, text: 'The sum over the trace is its description length. 512 tokens ≈ 126 bits = the model mostly predicted itself; the information lives in the spikes.' }
      ]
    },
    {
      id: 'ladder',
      title: 'classic ladder (logit lens)',
      items: [
        { ok: true, text: 'Shows WHERE in depth the final prediction crystallizes — good for comparing columns, prompts, models, checkpoints.' },
        { ok: false, text: 'A rung’s decode is not “what layer k believes.” The unembedding was only trained against the final layer; mid-rungs can decode junk (‘oret’, ‘oretical’) while the layer is doing useful work.' },
        { ok: false, text: 'Foreign-vocab tokens mid-ladder (二百 for “two hundred”) are nearest-neighbor artifacts of a language-agnostic concept space — not “thinking in Chinese.” Late layers choose surface form: language, register, spelling.' },
        { ok: false, text: 'Bits rising over the last layers is usually calibration/hedging, not forgetting (gpt2’s Paris: 2.0b at layer 9, 3.8b at final).' }
      ]
    },
    {
      id: 'jlens',
      title: 'J-lens',
      items: [
        { ok: true, text: 'The rung’s content transported through the remaining layers: “what does this layer contribute to the final answer.” Trust it mid-network, where the classic decode is illegible.' },
        { ok: true, text: 'Classic-minus-J gap = information already present but not yet rotated into the readout basis.' },
        { ok: false, text: 'The two curves ALWAYS meet at the final rung — J is the identity there by construction. Their meeting is not a finding.' },
        { ok: false, text: 'It is a linearization (one JVP): readings far from the final layer are extrapolations, and the embed rung’s J decode is usually junk. Exactly tied percentages near a fork are knife-edge numerics, not deep structure.' }
      ]
    },
    {
      id: 'attn',
      title: 'attention (purple shading, head grid)',
      items: [
        { ok: true, text: 'A hypothesis about where the computation looked. The default overlay is value-weighted (a·‖v‖), which already discounts stares at near-zero value vectors.' },
        { ok: false, text: 'Attention weight ≠ importance (“attention is not explanation”). To claim a region mattered, ablate it and read Δbits — weights only nominate candidates.' },
        { ok: false, text: 'Mass on <|im_start|> is an attention sink — a parking spot, not interest. Previous-token and local stripes are positional plumbing. Final-layer heads are mostly retired.' },
        { ok: false, text: 'Diffuse ≠ integrating, focused ≠ important. Before naming a head, click it and see whether its value-weighted gaze concentrates or stays smeared (mean-pooling wears a busy expression).' }
      ]
    },
    {
      id: 'ablate',
      title: 'Δbits ablation (re-price without a region)',
      items: [
        { ok: true, text: 'Causal, in bits: the delta is what reading that region actually bought for THIS token. This is the arbiter the attention overlay defers to.' },
        { ok: false, text: 'Δ≈0 does not mean the reasoning was useless — the visible context may re-derive it. A shown-work answer priced 408 at only +0.11b without the entire think block, because “200+40+140+28 =” was still on screen.' },
        { ok: false, text: 'Masking is off-distribution surgery: a huge mask can derail the computation rather than cleanly remove information. Prefer tight regions and compare against masking a same-sized neutral region.' }
      ]
    },
    {
      id: 'repro',
      title: 'sampling & reproducibility',
      items: [
        { ok: true, text: 'Same seed + temperature replays a trace exactly — within one engine session.' },
        { ok: false, text: 'Across engine restarts, near-tie forks can flip (bf16 kernels aren’t bitwise stable) and the trace diverges from there. Drill-ins use exact token ids, so they never fork.' },
        { ok: true, text: 'Ladder differences under ~0.3b between runs are numeric noise. Don’t interpret them.' }
      ]
    },
    {
      id: 'small',
      title: 'small-model humility',
      items: [
        { ok: false, text: 'This is a 0.6B model: it ruminates under greedy decoding, confabulates details (“the one-child policy” as a protest cause), and holds instructions loosely against trained habits (\\boxed{} beat “just the number”).' },
        { ok: true, text: 'One column is an anecdote. Before concluding, run a control: another topic in the same frame, another seed, or base vs chat on matched context — the deltas are the evidence.' }
      ]
    }
  ];

  let { sections }: { sections: string[] } = $props();
  const shown = $derived(ALL.filter((s) => sections.includes(s.id)));
</script>

<div class="guide scrollbar">
  {#each shown as s (s.id)}
    <div class="gsec">
      <div class="gtitle mono">{s.title}</div>
      {#each s.items as it, i (i)}
        <div class="gitem" class:bad={!it.ok}>
          <span class="mark mono">{it.ok ? '✓' : '✗'}</span>
          <span class="gtext">{it.text}</span>
        </div>
      {/each}
    </div>
  {/each}
</div>

<style>
  .guide {
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow: auto;
    min-height: 0;
    font-size: 11.5px;
    line-height: 1.55;
  }
  .gsec { display: flex; flex-direction: column; gap: 5px; }
  .gtitle {
    font-size: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border-2);
    padding-bottom: 3px;
  }
  .gitem { display: flex; gap: 8px; align-items: baseline; }
  .mark { flex: 0 0 auto; color: var(--ok, #4dc07d); font-size: 11px; }
  .gitem.bad .mark { color: var(--bad, #e5484d); }
  .gtext { color: var(--text, inherit); }
  .gitem.bad .gtext { color: var(--muted); }
</style>
