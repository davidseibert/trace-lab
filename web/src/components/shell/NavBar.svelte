<script lang="ts">
  /**
   * The shell-owned nav: wordmark → index, plus the lens tabs grouped by kind
   * (instruments / toys / concepts), all rendered from the registry. Lenses no
   * longer receive a `brand` snippet — navigation belongs to the shell.
   */
  import { router } from '../../lib/router.svelte';
  import { KINDS, KIND_ORDER, LENSES } from '../../lib/lenses';
</script>

<header class="nav panel">
  <a class="wordmark mono" href="#/" title="index — all lenses">trace<span class="dim">·lab</span></a>
  <a class="math mono" class:active={router.path === 'math'} href="#/math" title="Derive the mathematics behind every readout">∑ Math</a>

  {#each KIND_ORDER as k (k)}
    <div class="group" title={KINDS[k].tagline}>
      <span class="glabel">{KINDS[k].label}</span>
      <nav class="tabs">
        {#each LENSES.filter((l) => l.kind === k) as l (l.id)}
          <a class="tab mono" class:active={router.path === l.id} href={`#/${l.id}`} title={l.blurb}>
            {l.title}
          </a>
        {/each}
      </nav>
    </div>
  {/each}
</header>

<style>
  .nav {
    display: flex;
    flex: 0 0 auto;
    align-items: center;
    gap: 18px;
    padding: 6px 12px;
    flex-wrap: wrap;
  }
  .wordmark {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.01em;
    white-space: nowrap;
    color: var(--text);
    text-decoration: none;
  }
  .wordmark .dim { color: var(--faint); font-weight: 500; }
  .math {
    color: var(--model); text-decoration: none; white-space: nowrap;
    border: 1px solid var(--border-2); border-radius: var(--r-sm);
    padding: 3px 8px; font-size: 11px;
  }
  .math:hover, .math.active { border-color: var(--model); background: rgba(91, 156, 255, 0.12); }

  .group { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .glabel {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--faint);
    white-space: nowrap;
  }
  .tabs {
    display: inline-flex;
    border: 1px solid var(--border-2);
    border-radius: var(--r-sm);
    overflow: hidden;
  }
  .tab {
    padding: 4px 10px;
    font-size: 12px;
    white-space: nowrap;
    color: var(--text);
    text-decoration: none;
  }
  .tab:hover { background: #232a3a; }
  .tab.active { background: var(--model); color: #06121f; font-weight: 600; }
</style>
