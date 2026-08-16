<script lang="ts">
  /**
   * The shell: NavBar (registry-driven, grouped by kind) + whatever the hash
   * routes to. Lens metadata lives in lib/lenses.ts; this file only maps ids
   * to components, so the registry stays free of component imports (and thus
   * of import cycles — components import the registry for titles/links).
   */
  import type { Component } from 'svelte';
  import { router } from './lib/router.svelte';
  import { lensById } from './lib/lenses';

  import NavBar from './components/shell/NavBar.svelte';
  import IndexPage from './components/shell/IndexPage.svelte';
  import GrammarApp from './components/GrammarApp.svelte';
  import MorphApp from './components/MorphApp.svelte';
  import MorfessorApp from './components/MorfessorApp.svelte';
  import LlmApp from './components/LlmApp.svelte';
  import GraphApp from './components/GraphApp.svelte';
  import CoderApp from './components/CoderApp.svelte';
  import LogitApp from './components/LogitApp.svelte';
  import ReasonApp from './components/ReasonApp.svelte';
  import AttentionLabApp from './components/attn/AttentionLabApp.svelte';
  import HopfieldApp from './components/hopfield/HopfieldApp.svelte';
  import TictacApp from './components/tictac/TictacApp.svelte';
  import HopfieldRealApp from './components/hopfield/HopfieldRealApp.svelte';
  import TrainApp from './components/train/TrainApp.svelte';

  const COMPONENTS: Record<string, Component> = {
    grammar: GrammarApp,
    morph: MorphApp,
    morfessor: MorfessorApp,
    llm: LlmApp,
    attn: AttentionLabApp,
    hopfield: HopfieldApp,
    tictac: TictacApp,
    hopfieldreal: HopfieldRealApp,
    graph: GraphApp,
    coder: CoderApp,
    logit: LogitApp,
    reason: ReasonApp,
    train: TrainApp
  };

  // Unknown paths fall through to the index rather than a 404 — this is a
  // sandbox, not a site.
  const Active = $derived(lensById(router.path) ? COMPONENTS[router.path] : null);
</script>

<div class="app">
  <NavBar />
  {#if Active}
    {#key `${router.path}@${router.epoch}`}
      <Active />
    {/key}
  {:else}
    <IndexPage />
  {/if}
</div>

<style>
  .app {
    /* 100% (not 100dvh): the html→body→#app chain is 100%, and an independent
       dvh measure can exceed it, pushing the transport bar below the fold. */
    height: 100%;
    min-width: 0;
    overflow: clip;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
</style>
