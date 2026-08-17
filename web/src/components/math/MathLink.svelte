<script lang="ts">
  let {
    lesson,
    label = 'math behind this'
  }: {
    lesson: string;
    label?: string;
  } = $props();
</script>

<a class="math-link mono" href={`#/math?lesson=${lesson}`} title="Open the derivation in Math·foundations">
  ∑ {label}
</a>

<style>
  .math-link {
    color: var(--model);
    text-decoration: none;
    white-space: nowrap;
    font-size: 10.5px;
    border: 1px solid var(--border-2);
    border-radius: var(--r-sm);
    padding: 3px 7px;
  }
  .math-link:hover { border-color: var(--model); background: var(--panel-2); }
</style>
