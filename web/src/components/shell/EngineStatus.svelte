<script lang="ts">
  /**
   * The engine status pill plus the mount-time health probe, shared by every
   * lens that talks to the local engine service. The compact default renders
   * the topbar span every instrument lens uses; `detailed` (the index page)
   * also shows the pre-probe "checking…" state and the `make up` hint.
   * `onProbe` receives the probe result — Logit·real uses it to auto-run its
   * restored prompt when the engine answers.
   */
  import { engine } from '../../lib/logit/engine.svelte';

  let {
    detailed = false,
    onProbe
  }: {
    detailed?: boolean;
    onProbe?: (ok: boolean) => void;
  } = $props();

  $effect(() => {
    engine.check().then((ok) => onProbe?.(ok));
  });
</script>

{#if detailed}
  <span class="estatus mono" class:off={engine.probed && !engine.up}>
    {!engine.probed ? 'engine — checking…' : engine.up ? `engine · ${engine.device}` : 'engine offline — make up'}
  </span>
{:else}
  <span class="status mono" class:off={!engine.up} title="engine service (engine/, port 5181)">
    {engine.up ? `engine · ${engine.device}` : 'engine offline'}
  </span>
{/if}

<style>
  /* .status / .status.off are global topbar vocabulary (app.css). */
  .estatus { font-size: 10.5px; color: var(--good); margin-left: auto; white-space: nowrap; }
  .estatus.off { color: var(--bad); }
</style>
