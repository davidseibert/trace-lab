<script lang="ts">
  /**
   * The front door at #/ — the registry rendered as three sections with
   * different weight: instruments lead (open-ended workbenches, live engine
   * status), then the toys, then the concept exhibits. Cards are just links;
   * every lens keeps its state in its own URL.
   */
  import { KINDS, KIND_ORDER, LENSES, type LensKind } from '../../lib/lenses';
  import { engine } from '../../lib/logit/engine.svelte';

  $effect(() => {
    void engine.check();
  });

  const byKind = (k: LensKind) => LENSES.filter((l) => l.kind === k);
</script>

<div class="index scrollbar">
  <div class="hero">
    <p class="thesis">
      <span class="mono formula">
        <b style="color:var(--total)">min</b>
        <b style="color:var(--model)">L(M)</b> + <b style="color:var(--data)">L(D|M)</b>
      </span>
      — the model that predicts best is the model that compresses best. Every lens below turns some
      corner of that sentence into a falling bits curve you can scrub.
    </p>
  </div>

  {#each KIND_ORDER as k (k)}
    <section class="tier" class:instrument={k === 'instrument'}>
      <div class="tier-head">
        <h2 class="mono">{KINDS[k].label}</h2>
        <span class="tagline faint">{KINDS[k].tagline}</span>
        {#if k === 'instrument'}
          <span class="estatus mono" class:off={engine.probed && !engine.up}>
            {!engine.probed ? 'engine — checking…' : engine.up ? `engine · ${engine.device}` : 'engine offline — make up'}
          </span>
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

  .hero { padding: 4px 8px 0; max-width: 760px; }
  .thesis { margin: 0; font-size: 13.5px; line-height: 1.6; color: var(--muted); }
  .formula { font-size: 13px; }

  .tier { display: flex; flex-direction: column; gap: 10px; }
  .tier-head { display: flex; align-items: baseline; gap: 12px; padding: 0 8px; flex-wrap: wrap; }
  .tier-head h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text);
  }
  .tagline { font-size: 11.5px; }
  .estatus { font-size: 10.5px; color: var(--good); margin-left: auto; white-space: nowrap; }
  .estatus.off { color: var(--bad); }

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
