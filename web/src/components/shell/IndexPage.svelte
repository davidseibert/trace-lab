<script lang="ts">
  /**
   * The front door at #/ — the registry rendered as three sections with
   * different weight: instruments lead (open-ended workbenches, live engine
   * status), then the toys, then the concept exhibits. Cards are just links;
   * every lens keeps its state in its own URL.
   */
  import { KINDS, KIND_ORDER, LENSES, type LensKind } from '../../lib/lenses';
  import { engine } from '../../lib/logit/engine.svelte';
  import EngineStatus from './EngineStatus.svelte';

  const byKind = (k: LensKind) => LENSES.filter((l) => l.kind === k);
</script>

<div class="index scrollbar">
  <div class="hero">
    <p class="thesis">
      <span class="mono formula">
        <b style="color:var(--total)">min</b>
        <b style="color:var(--model)">L(M)</b> + <b style="color:var(--data)">L(D|M)</b>
      </span>
      — under one probability model, prediction cost and ideal compression cost are the same
      <span class="mono">−log₂p</span>. Every lens below turns some corner of that statement into a falling bits curve you can scrub.
    </p>
    <a class="math-card panel" href="#/math">
      <span class="mono sigma">∑</span>
      <span><b>Math·foundations</b><small>Why bits? Derive surprisal, entropy, code lengths, and cross-entropy from ordinary counting.</small></span>
      <span class="arrow">→</span>
    </a>
  </div>

  {#each KIND_ORDER as k (k)}
    <section class="tier" class:instrument={k === 'instrument'}>
      <div class="tier-head">
        <h2 class="mono">{KINDS[k].label}</h2>
        <span class="tagline faint">{KINDS[k].tagline}</span>
        {#if k === 'instrument'}
          <EngineStatus detailed />
        {/if}
      </div>
      <div class="cards">
        {#each byKind(k) as l (l.id)}
          <a class="card panel" href={`#/${l.id}`}>
            <span class="ctitle mono">{l.title}</span>
            <span class="cblurb">{l.blurb}</span>
            {#if l.engine}
              <span class="cfoot faint mono">{engine.up ? '● engine ready' : '○ needs engine'}</span>
            {/if}
          </a>
        {/each}
      </div>
    </section>
  {/each}
</div>

<style>
  .index {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    display: flex;
    flex-direction: column;
    gap: 22px;
    padding: 14px 4px 24px;
  }

  .hero { padding: 4px 8px 0; max-width: 860px; display: flex; flex-direction: column; gap: 12px; }
  .thesis { margin: 0; font-size: 13.5px; line-height: 1.6; color: var(--muted); }
  .formula { font-size: 13px; }
  .math-card { display: grid; grid-template-columns: 34px 1fr auto; gap: 10px; align-items: center; padding: 10px 12px; color: var(--text); text-decoration: none; border-left: 3px solid var(--total); }
  .math-card:hover { border-color: var(--model); background: var(--panel-2); }
  .sigma { color: var(--total); font-size: 23px; }
  .math-card b { display: block; font-size: 12.5px; }
  .math-card small { display: block; color: var(--muted); font-size: 11px; margin-top: 2px; }
  .math-card .arrow { color: var(--model); }

  .tier { display: flex; flex-direction: column; gap: 10px; }
  .tier-head { display: flex; align-items: baseline; gap: 12px; padding: 0 8px; flex-wrap: wrap; }
  .tier-head h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text);
  }
  .tagline { font-size: 11.5px; }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 10px;
  }
  .tier.instrument .cards { grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }

  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px 14px;
    text-decoration: none;
    color: var(--text);
    transition: border-color 0.12s, background 0.12s;
  }
  .card:hover { border-color: var(--model); background: var(--panel-2); }
  .tier.instrument .card { border-left: 3px solid var(--model); }

  .ctitle { font-size: 13.5px; font-weight: 700; }
  .cblurb { font-size: 12px; line-height: 1.5; color: var(--muted); }
  .cfoot { font-size: 10px; margin-top: auto; padding-top: 4px; }
</style>
