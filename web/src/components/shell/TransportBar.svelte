<script lang="ts">
  /**
   * The shared transport strip: playback controls + the status note. Pass
   * `converged` to tint the note green (an MDL search that ran out of paying
   * moves, a coder at the end of its stream, …). Omit `player` for a
   * note-only bar — a lens with nothing to scrub (Train·real) still ends in
   * the same strip as every other lens.
   */
  import type { Snippet } from 'svelte';
  import type { Player } from '../../lib/player.svelte';
  import Controls from '../Controls.svelte';

  let {
    player,
    note = '',
    converged = false,
    children
  }: {
    player?: Player<unknown>;
    note?: string;
    converged?: boolean;
    /** Optional rich note content; wins over `note` when provided. */
    children?: Snippet;
  } = $props();
</script>

<div class="panel transport-panel">
  {#if player}<Controls {player} />{/if}
  <div class="note mono" class:converged title={note}>
    {#if children}{@render children()}{:else}{note}{/if}
  </div>
</div>

<style>
  .transport-panel {
    display: flex;
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 16px;
    padding: 7px 12px;
  }
  .note {
    flex: 1;
    min-width: 0;
    font-size: 12px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: right;
  }
  .note.converged { color: var(--good); }
</style>
