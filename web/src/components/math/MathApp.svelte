<script lang="ts">
  import { PanelManager } from '../../lib/panels/panels.svelte';
  import { MATH_LESSONS, mathLessonById } from '../../lib/math/lessons';
  import { router } from '../../lib/router.svelte';
  import { lensById } from '../../lib/lenses';

  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import LessonView from './LessonView.svelte';
  import MathLab from './MathLab.svelte';

  const panels = new PanelManager(
    'math',
    [
      { id: 'lesson', title: 'Derivation', zoomable: true },
      { id: 'lab', title: 'Work the numbers' },
      { id: 'symbols', title: 'Symbols & units', fit: true },
      { id: 'map', title: 'Learning path' },
      { id: 'links', title: 'Seen in trace·lab', fit: true }
    ],
    {
      columns: [['lesson'], ['lab', 'symbols', 'map', 'links']],
      widths: [1.35, 1],
      weights: { lab: 1.8, map: 0.8 }
    }
  );

  const requested = router.get('lesson');
  const lesson = mathLessonById(requested) ?? MATH_LESSONS[0];
  const index = MATH_LESSONS.indexOf(lesson);
  const previous = MATH_LESSONS[index - 1];
  const next = MATH_LESSONS[index + 1];

  function go(id: string) {
    location.hash = `/math${id === MATH_LESSONS[0].id ? '' : `?lesson=${id}`}`;
  }
</script>

<TopBar {panels}>
  <span class="formula mono"><b style="color:var(--total)">Math·foundations</b></span>
  <span class="faint promise">follow every displayed number back to counting</span>

  <label class="f">
    <span class="lbl">lesson</span>
    <select value={lesson.id} onchange={(e) => go((e.currentTarget as HTMLSelectElement).value)}>
      {#each MATH_LESSONS as item, i (item.id)}
        <option value={item.id}>{i + 1}. {item.shortTitle}</option>
      {/each}
    </select>
  </label>

  <span class="spacer"></span>
  {#if previous}<a class="navbtn mono" href={`#/math${previous.id === MATH_LESSONS[0].id ? '' : `?lesson=${previous.id}`}`}>← {previous.shortTitle}</a>{/if}
  <span class="progress mono">{index + 1} / {MATH_LESSONS.length}</span>
  {#if next}<a class="navbtn mono" href={`#/math?lesson=${next.id}`}>{next.shortTitle} →</a>{/if}
</TopBar>

{#snippet aLesson()}<span class="mono">lesson {index + 1} · {lesson.question}</span>{/snippet}
{#snippet pLesson()}<LessonView {lesson} />{/snippet}

{#snippet aLab()}<span class="mono">change a value; predict before reading</span>{/snippet}
{#snippet pLab()}<MathLab lab={lesson.lab} />{/snippet}

{#snippet pSymbols()}
  <div class="symbols">
    {#each lesson.terms as term (term.symbol)}
      <div class="symbol-row">
        <b class="mono">{term.symbol}</b>
        <span>{term.meaning}</span>
        {#if term.unit}<em class="mono">{term.unit}</em>{/if}
      </div>
    {/each}
  </div>
{/snippet}

{#snippet pMap()}
  <nav class="path scrollbar" aria-label="Math foundations lessons">
    {#each MATH_LESSONS as item, i (item.id)}
      <a
        class:current={item.id === lesson.id}
        class:past={i < index}
        href={`#/math${item.id === MATH_LESSONS[0].id ? '' : `?lesson=${item.id}`}`}
      >
        <span class="step mono">{i + 1}</span>
        <span><b>{item.shortTitle}</b><small>{item.question}</small></span>
      </a>
    {/each}
  </nav>
{/snippet}

{#snippet pLinks()}
  <div class="seen">
    {#each lesson.seenIn as item (item.lens)}
      {@const meta = lensById(item.lens)}
      <a href={`#/${item.lens}`}>
        <b class="mono">{meta?.title ?? item.label}</b>
        <span>{item.note}</span>
      </a>
    {/each}
  </div>
{/snippet}

<PanelHost
  manager={panels}
  snippets={{ lesson: pLesson, lab: pLab, symbols: pSymbols, map: pMap, links: pLinks }}
  actions={{ lesson: aLesson, lab: aLab }}
/>

<style>
  .promise { font-size: 11.5px; }
  .navbtn { color: var(--model); text-decoration: none; font-size: 10.5px; white-space: nowrap; }
  .navbtn:hover { color: var(--text); }
  .progress { color: var(--faint); font-size: 10px; }
  .symbols { display: flex; flex-direction: column; gap: 5px; }
  .symbol-row { display: grid; grid-template-columns: minmax(70px, auto) 1fr auto; gap: 9px; align-items: baseline; font-size: 11.5px; }
  .symbol-row b { color: var(--data); }
  .symbol-row span { color: var(--muted); }
  .symbol-row em { color: var(--total); font-style: normal; font-size: 10px; }
  .path { display: flex; flex-direction: column; gap: 3px; overflow: auto; min-height: 0; }
  .path a { display: grid; grid-template-columns: 22px 1fr; gap: 7px; padding: 5px 7px; border-radius: var(--r-sm); color: var(--muted); text-decoration: none; }
  .path a:hover { background: var(--panel-2); color: var(--text); }
  .path a.current { background: rgba(91,156,255,.12); outline: 1px solid rgba(91,156,255,.35); color: var(--text); }
  .path a.past .step { background: var(--total); color: #062019; }
  .step { width: 20px; height: 20px; display: grid; place-items: center; border: 1px solid var(--border-2); border-radius: 50%; font-size: 9px; }
  .path b { display: block; font-size: 11px; }
  .path small { display: block; color: var(--faint); font-size: 9.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .seen { display: flex; flex-direction: column; gap: 5px; }
  .seen a { display: grid; grid-template-columns: 100px 1fr; gap: 8px; color: var(--text); text-decoration: none; font-size: 11px; }
  .seen a:hover b { color: var(--model); }
  .seen span { color: var(--muted); }
</style>
