import {makeScene2D, Rect, Txt} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  sequence,
  ThreadGenerator,
  waitFor,
  waitUntil,
} from '@motion-canvas/core';

import data from '../data/words.json';
import {C, FONT, rng, Vec} from '../lib/viz';

// How the GPU encodes a window (`src/gpu/word_groups.rs`): memory resets per word, so
// words are independent. Padding every word to the longest wastes cells; instead words
// are grouped by exact length, each group is one dense [words, len + 1] batch, and the
// groups run one after another.
const S = 26;
const GROUP_GAP = 56;
const GROUP_Y = -150;
const OUT_Y = 360;
const OUT_STEP = 70;

const ACTIVE = C.orange;
const DONE = '#4a4230';
const WASTE = '#e07a6f';
const WASTE_DONE = '#3a2224';
const GROUP_COLORS = [C.blue, C.orange, C.purple, C.green, '#e07a6f', '#d4d46f'];

const glyph = (b: number) => (b === 32 ? '·' : b === 10 ? '⏎' : b < 128 ? String.fromCharCode(b) : '•');

const words = data.words;
const N = words.length;
const tmaxOf = (i: number) => words[i].bytes.length + 1;
const TMAX = Math.max(...words.map((_, i) => tmaxOf(i)));
const realCells = words.reduce((n, _, i) => n + tmaxOf(i), 0);
const paddedCells = N * TMAX;

const lengths = [...new Set(words.map((_, i) => tmaxOf(i)))].sort((a, b) => a - b);
const groups = lengths.map(t => ({tmax: t, members: words.map((_, i) => i).filter(i => tmaxOf(i) === t)}));

// Padded rectangle, words in window order.
const padX = (c: number) => (c - (TMAX - 1) / 2) * S;
const padY = (r: number) => (r - (N - 1) / 2) * S + 20;

// Groups side by side, top aligned.
const groupWidth = groups.reduce((w, g) => w + g.tmax * S, 0) + (groups.length - 1) * GROUP_GAP;
const groupX0 = groups.map((_, gi) =>
  -groupWidth / 2 + groups.slice(0, gi).reduce((w, g) => w + g.tmax * S + GROUP_GAP, 0) + S / 2,
);

const r = rng(31);
const outVals = words.map(() => Array.from({length: 6}, () => (r() * 2 - 1) * 2.8));

export default makeScene2D(function* (view) {
  view.fill(C.bg);

  const title = createRef<Txt>();
  const caption = createRef<Txt>();
  const counter = createRef<Txt>();

  // cells[i][c]: word i, step c; the last step of every word is its [W].
  const cells = words.map((w, i) =>
    Array.from({length: tmaxOf(i)}, (_, c) => {
      const isW = c === tmaxOf(i) - 1;
      return new Rect({
        x: padX(c),
        y: padY(i),
        width: S - 4,
        height: S - 4,
        radius: 4,
        fill: C.panel,
        stroke: isW ? C.purple : C.wire,
        lineWidth: 2,
        opacity: 0,
        children: [
          new Txt({
            fontFamily: FONT,
            fontSize: isW ? 10 : 15,
            fill: isW ? C.purple : C.text,
            text: isW ? 'W' : glyph(w.bytes[c]),
          }),
        ],
      });
    }),
  );
  const padding = words.map((_, i) =>
    Array.from(
      {length: TMAX - tmaxOf(i)},
      (_, k) =>
        new Rect({
          x: padX(tmaxOf(i) + k),
          y: padY(i),
          width: S - 4,
          height: S - 4,
          radius: 4,
          stroke: C.wire,
          lineWidth: 2,
          lineDash: [4, 4],
          opacity: 0,
        }),
    ),
  );
  const outs = words.map(
    (_, i) => new Vec(outVals[i], {x: padX(tmaxOf(i) - 1), y: padY(i), step: 12, thickness: 30}),
  );

  view.add(
    <>
      <Txt ref={title} y={-470} fontFamily={FONT} fontSize={46} fill={C.text} opacity={0} />
      <Txt ref={counter} y={-395} fontFamily={FONT} fontSize={26} fill={C.dim} opacity={0} />
      <Txt ref={caption} y={490} fontFamily={FONT} fontSize={28} fill={C.dim} opacity={0} />
    </>,
  );
  view.add([...cells.flat(), ...padding.flat()]);
  view.add(outs.map(v => v.node));

  /** One time step over a set of rows: every row advances together. */
  function* step(rows: number[], c: number, withPadding: boolean, d: number): ThreadGenerator {
    const live = rows.filter(i => c < tmaxOf(i)).map(i => cells[i][c]);
    const pad = withPadding ? rows.filter(i => c >= tmaxOf(i)).map(i => padding[i][c - tmaxOf(i)]) : [];
    yield* all(
      ...live.map(cell => cell.fill(ACTIVE, 0.4 * d).to(DONE, 0.6 * d)),
      ...pad.map(cell => cell.fill(WASTE, 0.4 * d).to(WASTE_DONE, 0.6 * d)),
    );
  }

  const allRows = words.map((_, i) => i);

  // Every word is its own short sequence.
  title().text('encoding all words at once');
  yield* title().opacity(1, 0.5);
  yield* sequence(0.04, ...cells.map(row => all(...row.map(c => c.opacity(1, 0.3)))));
  caption().text('memory resets for every word: no word waits for another');
  yield* caption().opacity(1, 0.5);

  // One batch, padded to the longest word.
  yield* waitUntil('pad');
  yield* all(
    caption().text('one batch: pad every word to the longest', 0.6),
    ...padding.flat().map(p => p.opacity(1, 0.6)),
  );
  counter().text('steps: 0');
  yield* counter().opacity(1, 0.3);
  for (let c = 0; c < TMAX; c++) {
    counter().text(`steps: ${c + 1}`);
    yield* step(allRows, c, true, 0.7);
  }
  yield* caption().text(`${paddedCells - realCells} of ${paddedCells} cells are padding`, 0.6);
  yield* waitFor(1);

  // Group by exact length: no padding at all.
  yield* waitUntil('groups');
  yield* all(
    caption().text('group words by length instead', 0.6),
    counter().opacity(0, 0.3),
    ...padding.flat().map(p => p.opacity(0, 0.5)),
    ...cells.flat().map(c => c.fill(C.panel, 0.5)),
  );
  yield* all(
    ...groups.flatMap((g, gi) =>
      g.members.flatMap((i, k) =>
        cells[i].map((cell, c) =>
          all(
            cell.position([groupX0[gi] + c * S, GROUP_Y + k * S], 1.2, easeInOutCubic),
            c < g.tmax - 1 ? cell.stroke(GROUP_COLORS[gi % GROUP_COLORS.length], 1.2) : cell.stroke(C.purple, 0),
          ),
        ),
      ),
    ),
  );

  // Each group is one dense batch; groups run one after another.
  yield* waitUntil('run_groups');
  caption().text('each group is one dense batch, groups run one after another');
  counter().text('steps: 0');
  yield* counter().opacity(1, 0.3);
  let n = 0;
  for (const g of groups) {
    for (let c = 0; c < g.tmax; c++) {
      counter().text(`steps: ${++n}`);
      yield* step(g.members, c, false, 0.45);
    }
  }
  yield* caption().text(`${realCells} cells, zero padding · a real 4096-word window: 16 groups`, 0.6);
  yield* waitFor(1.2);

  // Word vectors go back into window order for the backbone.
  yield* waitUntil('scatter');
  const outX = (i: number) => (i - (N - 1) / 2) * OUT_STEP;
  for (const [gi, g] of groups.entries()) {
    g.members.forEach((i, k) => outs[i].node.position([groupX0[gi] + (g.tmax - 1) * S, GROUP_Y + k * S]));
  }
  outs.forEach(v => v.node.scale(0.3));
  yield* all(
    caption().text('the word vectors go back into window order', 0.6),
    counter().opacity(0, 0.4),
    ...cells.flat().map(c => c.opacity(0.3, 0.6)),
    ...outs.map((v, i) =>
      all(
        v.node.opacity(1, 0.3),
        v.node.position([outX(i), OUT_Y], 1.2, easeInOutCubic),
        v.node.scale(1, 1.2, easeInOutCubic),
      ),
    ),
  );
  yield* caption().text('→ backbone', 0.5);
  yield* waitFor(1.5);
});
