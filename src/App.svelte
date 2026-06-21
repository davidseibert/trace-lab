<script lang="ts">
  import GrammarApp from './components/GrammarApp.svelte';
  import MorphApp from './components/MorphApp.svelte';
  import MorfessorApp from './components/MorfessorApp.svelte';
  import LlmApp from './components/LlmApp.svelte';
  import GraphApp from './components/GraphApp.svelte';
  import CoderApp from './components/CoderApp.svelte';

  type Lens = 'grammar' | 'morph' | 'morfessor' | 'llm' | 'graph' | 'coder';
  let lens = $state<Lens>('grammar');
</script>

<!-- Shared identity + lens switcher, handed to whichever lens is active so the
     whole app keeps a single top bar. -->
{#snippet brand()}
  <div class="brand">
    <span class="wordmark mono">trace<span class="dim">·lab</span></span>
    <div class="toggle-group lens-switch">
      <button class:active={lens === 'grammar'} onclick={() => (lens = 'grammar')}>MDL Grammar</button>
      <button class:active={lens === 'morph'} onclick={() => (lens = 'morph')}>Morph·merge</button>
      <button class:active={lens === 'morfessor'} onclick={() => (lens = 'morfessor')}>Morph·split</button>
      <button class:active={lens === 'llm'} onclick={() => (lens = 'llm')}>Mini-GPT</button>
      <button class:active={lens === 'graph'} onclick={() => (lens = 'graph')}>Graph·SUBDUE</button>
      <button class:active={lens === 'coder'} onclick={() => (lens = 'coder')}>Coder</button>
    </div>
  </div>
{/snippet}

<div class="app">
  {#if lens === 'grammar'}
    <GrammarApp {brand} />
  {:else if lens === 'morph'}
    <MorphApp {brand} />
  {:else if lens === 'morfessor'}
    <MorfessorApp {brand} />
  {:else if lens === 'llm'}
    <LlmApp {brand} />
  {:else if lens === 'graph'}
    <GraphApp {brand} />
  {:else}
    <CoderApp {brand} />
  {/if}
</div>

<style>
  .app {
    height: 100dvh;
    max-width: 1500px;
    margin: 0 auto;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .brand { display: flex; align-items: center; gap: 10px; }
  .wordmark { font-size: 15px; font-weight: 700; letter-spacing: -0.01em; white-space: nowrap; }
  .wordmark .dim { color: var(--faint); font-weight: 500; }
  .lens-switch button { padding: 4px 10px; font-size: 12px; white-space: nowrap; }
</style>
