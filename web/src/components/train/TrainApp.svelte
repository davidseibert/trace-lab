<script lang="ts">
  /**
   * Train·real — the process behind `make train`, surfaced as a lens instead
   * of hidden in a script. Streams the engine's /train SSE feed: loss and
   * held-out accuracy per epoch, checkpoints announced as they land. The
   * checkpoints ARE the artifacts: each row links into Logit·real with a
   * prompt that fits the tiny model, and /health re-lists them on completion.
   */
  import { streamTrain, type TrainStart } from '../../lib/logit/api';
  import { engine } from '../../lib/logit/engine.svelte';
  import { router } from '../../lib/router.svelte';
  import { PanelManager } from '../../lib/panels/panels.svelte';

  import InterpretGuide from '../InterpretGuide.svelte';
  import PanelHost from '../PanelHost.svelte';
  import TopBar from '../shell/TopBar.svelte';
  import TransportBar from '../shell/TransportBar.svelte';
  import EngineOffline from '../shell/EngineOffline.svelte';
  import EngineStatus from '../shell/EngineStatus.svelte';
  import TrainChart from './TrainChart.svelte';

  const panels = new PanelManager(
    'train',
    [
      { id: 'curve', title: 'Loss & held-out accuracy' },
      { id: 'ckpts', title: 'Checkpoints' },
      { id: 'about', title: 'The task' },
      { id: 'guide', title: 'How to read this', collapsed: true }
    ],
    {
      columns: [['curve'], ['ckpts', 'about', 'guide']],
      widths: [2, 1.2],
      weights: { curve: 2, ckpts: 1.2 }
    }
  );

  let epochs = $state(router.num('ep') ?? 40);

  $effect(() => {
    router.setQuery({ ep: epochs === 40 ? null : epochs });
  });

  let streaming = $state(false);
  let error = $state('');
  let start = $state<TrainStart | null>(null);
  let rows = $state<{ epoch: number; loss: number; acc: number }[]>([]);
  let batch = $state<{ epoch: number; frac: number; loss: number } | null>(null);
  let ckpts = $state<{ name: string; note: string }[]>([]);
  let done = $state<{ acc: number; epochs_run: number } | null>(null);
  let abort: AbortController | null = null;

  async function run() {
    if (streaming) return;
    streaming = true;
    error = '';
    start = null;
    rows = [];
    batch = null;
    ckpts = [];
    done = null;
    abort = new AbortController();
    try {
      await streamTrain(
        { epochs },
        (ev) => {
          if (ev.event === 'start') start = ev;
          else if (ev.event === 'batch') batch = ev;
          else if (ev.event === 'epoch') {
            rows = [...rows, { epoch: ev.epoch, loss: ev.loss, acc: ev.acc }];
            batch = null;
          } else if (ev.event === 'checkpoint') {
            ckpts = [...ckpts.filter((c) => c.name !== ev.name), { name: ev.name, note: ev.note }];
          } else if (ev.event === 'done') done = { acc: ev.acc, epochs_run: ev.epochs_run };
          else error = ev.detail;
        },
        abort.signal
      );
      engine.markUp();
      // Freshly-trained checkpoints: re-probe so Logit·real's picker sees them.
      await engine.check();
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        error = e instanceof Error ? e.message : String(e);
        await engine.check();
      }
    } finally {
      streaming = false;
      abort = null;
    }
  }

  function stop() {
    abort?.abort();
  }

  // Where each checkpoint opens in Logit·real: pinned model + a prompt that
  // actually fits the 16-position, digits-only tokenizer.
  const lensLink = (name: string) =>
    `#/logit?m=${encodeURIComponent(`local/${name}`)}&p=${encodeURIComponent('17+25=')}`;

  const note = $derived(
    done
      ? `done — ${(done.acc * 100).toFixed(1)}% held-out after ${done.epochs_run} epochs`
      : streaming
        ? batch
          ? `epoch ${batch.epoch} · ${(batch.frac * 100).toFixed(0)}% · loss ${batch.loss.toFixed(4)}`
          : rows.length
            ? `epoch ${rows.length} scored`
            : 'initializing…'
        : ''
  );
</script>

<TopBar {panels}>
  <span class="formula mono" title="training minimizes L(D|M) — watch it fall live">
    <b style="color:var(--data)">L(D|M)</b> ↓ · training
  </span>

  <span class="task mono muted">tiny GPT-2 · 2-digit addition · loss on the 3 answer digits</span>

  <label class="f" title="upper bound — stops early at ≥99.9% held-out">
    <span class="lbl">epochs</span>
    <input class="ep mono" type="number" min="1" max="200" bind:value={epochs} />
  </label>

  {#if streaming}
    <button class="primary" onclick={stop}>stop ■</button>
  {:else}
    <button class="primary" onclick={run} disabled={!engine.up}>train ▸</button>
  {/if}

  <span class="spacer"></span>

  <EngineStatus />
</TopBar>

{#if !engine.up && !start}
  <EngineOffline
    what="This lens trains the tiny addition model on the local engine service."
  />
{:else}
  {#snippet aCurve()}
    {#if start}
      <span class="mono">{start.params_m}M params · {start.device} · {start.n_train} train / {start.n_test} held-out</span>
    {/if}
  {/snippet}
  {#snippet pCurve()}
    <TrainChart {rows} />
  {/snippet}

  {#snippet aCkpts()}
    <span class="mono">{ckpts.length} saved</span>
  {/snippet}
  {#snippet pCkpts()}
    <div class="ckpts scrollbar">
          {#if ckpts.length === 0}
            <p class="faint">
              Three checkpoints land here as training runs — untrained, first ≥50%, converged —
              and appear in the engine's model list as <span class="mono">local/add-*</span>.
            </p>
          {/if}
          {#each ckpts as c (c.name)}
            <div class="ckpt">
              <div class="ckhead">
                <span class="ckname mono">local/{c.name}</span>
                <a class="cklink mono" href={lensLink(c.name)} title="Open this checkpoint in Logit·real on a prompt that fits it">
                  Logit·real →
                </a>
              </div>
              <div class="cknote faint">{c.note}</div>
            </div>
          {/each}
          {#if done}
            <p class="donenote good mono">
              ✓ training finished — the checkpoints are live in every model picker.
            </p>
          {/if}
        </div>
  {/snippet}

  {#snippet pAbout()}
    <div class="about scrollbar">
          <p>
            Every <span class="mono">"a+b="</span> for a, b in 0–99, spelled one character per
            token over the vocabulary <span class="mono">0123456789+=</span>. Loss lands only on
            the three answer digits, so the model must route both addends — carry included —
            through attention. 8,000 sums train; 2,000 are held out.
          </p>
          <p class="muted">
            This is the same run as <span class="mono">make train</span>, just visible: the
            training-time axis that Logit·real's J-lens can then walk across checkpoints.
          </p>
        </div>
  {/snippet}

  {#snippet pGuide()}
    <InterpretGuide lens="train" sections={['trainread']} />
  {/snippet}

  <PanelHost
    manager={panels}
    snippets={{ curve: pCurve, ckpts: pCkpts, about: pAbout, guide: pGuide }}
    actions={{ curve: aCurve, ckpts: aCkpts }}
  />

  <TransportBar note={error ? `error · ${error}` : note || 'ready — press train'} />
{/if}

<style>
  .task { font-size: 11px; white-space: nowrap; }
  .ep { width: 56px; }

  .ckpts { display: flex; flex-direction: column; gap: 10px; overflow: auto; min-height: 0; }
  .ckpts p { margin: 0; font-size: 12px; line-height: 1.5; }
  .ckpt { display: flex; flex-direction: column; gap: 2px; border-left: 2px solid var(--model); padding-left: 8px; }
  .ckhead { display: flex; align-items: baseline; gap: 10px; }
  .ckname { font-size: 12px; font-weight: 600; }
  .cklink { font-size: 11px; color: var(--model); text-decoration: none; margin-left: auto; white-space: nowrap; }
  .cklink:hover { text-decoration: underline; }
  .cknote { font-size: 11px; line-height: 1.4; }
  .donenote { font-size: 11.5px; }

  .about { display: flex; flex-direction: column; gap: 8px; overflow: auto; min-height: 0; }
  .about p { margin: 0; font-size: 12.5px; line-height: 1.5; }
</style>
