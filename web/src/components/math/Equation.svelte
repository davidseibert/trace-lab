<script lang="ts">
  let { title, lines, note }: { title?: string; lines: string[]; note?: string } = $props();
</script>

<figure class="equation">
  {#if title}<figcaption>{title}</figcaption>{/if}
  <div class="elines mono">
    {#each lines as line}<div>{line}</div>{/each}
  </div>
  {#if note}<p>{note}</p>{/if}
</figure>

<style>
  .equation {
    margin: 0;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--model);
    border-radius: var(--r-sm);
    background: var(--bg-2);
  }
  figcaption {
    margin-bottom: 6px;
    color: var(--muted);
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }
  .elines { display: flex; flex-direction: column; gap: 4px; color: var(--text); line-height: 1.55; }
  p { margin: 7px 0 0; color: var(--muted); font-size: 11.5px; }
</style>
