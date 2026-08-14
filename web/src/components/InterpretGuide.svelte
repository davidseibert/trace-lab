<script lang="ts">
  /**
   * Pure renderer for the ✓/✗ field guides in lib/guides.ts. Every lens mounts
   * one inside a default-collapsed "How to read this" panel; passing the lens
   * id also renders its registry cross-links, so instruments point down at the
   * concept/toy that isolates their mechanism.
   */
  import { GUIDE_SECTIONS } from '../lib/guides';
  import { lensById } from '../lib/lenses';

  let { sections, lens }: { sections: string[]; lens?: string } = $props();
  const shown = $derived(sections.map((id) => GUIDE_SECTIONS[id]).filter(Boolean));
  const related = $derived(
    (lens ? (lensById(lens)?.seeAlso ?? []) : [])
      .map((id) => lensById(id))
      .filter((m) => m !== undefined)
  );
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

  {#if related.length}
    <div class="gsec">
      <div class="gtitle mono">see also</div>
      <div class="rel">
        {#each related as r (r.id)}
          <a class="mono" href={`#/${r.id}`} title={r.blurb}>{r.title}</a>
        {/each}
      </div>
    </div>
  {/if}
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

  .rel { display: flex; flex-wrap: wrap; gap: 6px; }
  .rel a {
    font-size: 11px;
    color: var(--model);
    text-decoration: none;
    border: 1px solid var(--border-2);
    border-radius: var(--r-sm);
    padding: 2px 8px;
  }
  .rel a:hover { border-color: var(--model); }
</style>
