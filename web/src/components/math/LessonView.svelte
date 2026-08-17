<script lang="ts">
  import type { MathLesson } from '../../lib/math/types';
  import { mathLessonById } from '../../lib/math/lessons';
  import Equation from './Equation.svelte';

  let { lesson }: { lesson: MathLesson } = $props();
  const prerequisites = $derived(
    lesson.prerequisites.map((id) => mathLessonById(id)).filter((x) => x !== undefined)
  );
</script>

<article class="lesson scrollbar">
  <header>
    <span class="eyebrow mono">{lesson.chapter}</span>
    <h1>{lesson.title}</h1>
    <p class="question">{lesson.question}</p>
    <p class="summary">{lesson.summary}</p>
    {#if prerequisites.length}
      <div class="prereqs">
        <span class="faint">builds on</span>
        {#each prerequisites as p (p.id)}
          <a class="mono" href={`#/math?lesson=${p.id}`}>{p.shortTitle}</a>
        {/each}
      </div>
    {/if}
  </header>

  {#each lesson.blocks as block, i (i)}
    {#if block.kind === 'equation'}
      <Equation title={block.title} lines={block.lines} note={block.note} />
    {:else if block.kind === 'list'}
      <section>
        {#if block.title}<h2>{block.title}</h2>{/if}
        <ul>{#each block.items as item}<li>{item}</li>{/each}</ul>
      </section>
    {:else if block.kind === 'callout'}
      <aside class="callout {block.tone ?? 'plain'}">
        <h2>{block.title}</h2>
        <p>{block.text}</p>
      </aside>
    {:else}
      <section>
        {#if block.title}<h2>{block.title}</h2>{/if}
        <p>{block.text}</p>
      </section>
    {/if}
  {/each}
</article>

<style>
  .lesson { overflow: auto; min-height: 0; padding: 2px 8px 22px 2px; line-height: 1.65; }
  header { border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 16px; }
  .eyebrow { color: var(--total); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; }
  h1 { margin-top: 4px; font-size: 24px; letter-spacing: -0.02em; }
  .question { margin: 6px 0 0; color: var(--data); font-size: 14px; font-weight: 600; }
  .summary { margin: 10px 0 0; color: var(--muted); font-size: 13.5px; max-width: 74ch; }
  .prereqs { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 10px; font-size: 10.5px; }
  .prereqs a { color: var(--model); text-decoration: none; border: 1px solid var(--border-2); border-radius: 999px; padding: 1px 7px; }
  section { margin: 0 0 16px; max-width: 78ch; }
  h2 { margin: 0 0 5px; color: var(--text); font-size: 12px; letter-spacing: 0.02em; }
  p { margin: 0; }
  ul { margin: 4px 0 0; padding-left: 20px; }
  li + li { margin-top: 3px; }
  :global(.equation) { margin-bottom: 16px; }
  .callout { padding: 10px 12px; border-radius: var(--r-sm); background: var(--panel-2); border-left: 3px solid var(--model); }
  .callout.good { border-left-color: var(--good); }
  .callout.warn { border-left-color: var(--chosen); }
  .callout p { color: var(--muted); }
</style>
