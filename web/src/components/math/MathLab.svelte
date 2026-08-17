<script lang="ts">
  import type { MathLabId } from '../../lib/math/types';
  import ProbabilityBitsLab from './labs/ProbabilityBitsLab.svelte';
  import EntropyLab from './labs/EntropyLab.svelte';
  import ArithmeticLab from './labs/ArithmeticLab.svelte';
  import CrossEntropyLab from './labs/CrossEntropyLab.svelte';

  let { lab }: { lab: MathLabId } = $props();
</script>

{#if lab === 'entropy'}
  <EntropyLab />
{:else if lab === 'arithmetic'}
  <ArithmeticLab />
{:else if lab === 'cross-entropy'}
  <CrossEntropyLab />
{:else}
  <ProbabilityBitsLab />
{/if}
