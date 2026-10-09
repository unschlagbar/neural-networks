import {Line, makeScene2D, Node, Rect, Txt} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  ThreadGenerator,
  waitFor,
  waitUntil,
} from '@motion-canvas/core';

import data from '../data/words.json';
import {C, FONT, rng, Vec} from '../lib/viz';

// The encoder from `src/model.rs`: byte embedding → sLSTM → mLSTM → sLSTM, run over
// the word's bytes plus a closing [W] step. The word vector is the output at [W].
const WORD = 'println';
const LEVELS = [
  {name: 'sLSTM', color: C.blue, y: 150},
  {name: 'mLSTM', color: C.orange, y: 30},
  {name: 'sLSTM', color: C.blue, y: -90},
];
const HIDDEN = 6;
const COL0 = -560;
const COL_STEP = 180;
const CHIP_Y = 390;
const EMB_Y = 270;
const TOP_Y = -290;
const LABEL_X = -790;
const ROW_STEP = 230;
const ROW_CHIP_Y = 220;
const ROW_VEC_Y = -30;
const VEC = {step: 18, thickness: 18, horizontal: true};

const glyph = (s: string) => s.replace(/ /g, '·').replace(/\n/g, '⏎');

const wordIndex = data.words.findIndex(w => w.text === WORD);
const word = data.words[wordIndex];
const steps = [
  ...[...word.text].map((ch, i) => ({label: ch, sub: String(word.bytes[i])})),
  {label: '[W]', sub: 'end'},
];
const T = steps.length;
const colX = (t: number) => COL0 + t * COL_STEP;
const row = data.words.slice(wordIndex, wordIndex + 7);

// Illustrative values: the shapes are the real ones, the numbers are not.
const r = rng(23);
const rand = (scale: number) => Array.from({length: HIDDEN}, () => (r() * 2 - 1) * scale);
const embVals = steps.map(() => rand(2.2));
const outVals = LEVELS.map(() => steps.map(() => rand(2.8)));
const rowVals = row.map(() => rand(2.8));

export default makeScene2D(function* (view) {
  view.fill(C.bg);

  const title = createRef<Txt>();
  const caption = createRef<Txt>();
  const grid = createRef<Node>();
  const wordLabel = createRef<Txt>();

  const chips = steps.map(
    (s, t) =>
      new Rect({
        x: colX(t),
        y: CHIP_Y,
        width: s.label.length > 1 ? 84 : 64,
        height: 64,
        radius: 10,
        fill: C.panel,
        stroke: t === T - 1 ? C.purple : C.wire,
        lineWidth: 3,
        opacity: 0,
        children: [
          new Txt({fontFamily: FONT, fontSize: 32, fill: t === T - 1 ? C.purple : C.text, text: s.label}),
          new Txt({y: 56, fontFamily: FONT, fontSize: 16, fill: C.dim, text: s.sub}),
        ],
      }),
  );

  const bands = LEVELS.map(
    l =>
      new Rect({
        x: (colX(0) + colX(T - 1)) / 2,
        y: l.y,
        width: colX(T - 1) - colX(0) + 190,
        height: 70,
        radius: 12,
        fill: C.panel,
        stroke: l.color,
        lineWidth: 2,
        opacity: 0,
      }),
  );
  const labels = [
    new Txt({x: LABEL_X, y: EMB_Y, fontFamily: FONT, fontSize: 22, fill: C.dim, text: 'embedding', opacity: 0}),
    ...LEVELS.map(
      l => new Txt({x: LABEL_X, y: l.y, fontFamily: FONT, fontSize: 24, fill: l.color, text: l.name, opacity: 0}),
    ),
  ];

  // Memory carried from one letter to the next, per block.
  const arrows = LEVELS.map(l =>
    steps.slice(1).map(
      (_, t) =>
        new Line({
          points: [
            [colX(t) + 62, l.y],
            [colX(t + 1) - 62, l.y],
          ],
          stroke: l.color,
          lineWidth: 3,
          endArrow: true,
          arrowSize: 10,
          end: 0,
        }),
    ),
  );

  const emb = steps.map((_, t) => new Vec(embVals[t], {x: colX(t), y: EMB_Y, ...VEC}));
  // Each output starts with the colours of what feeds it and recolours on the way up.
  const outs = LEVELS.map((_, l) =>
    steps.map((_, t) => new Vec(l === 0 ? embVals[t] : outVals[l - 1][t], {x: colX(t), y: LEVELS[l].y, ...VEC})),
  );
  const wordVec = new Vec(outVals[LEVELS.length - 1][T - 1], {x: colX(T - 1), y: LEVELS[LEVELS.length - 1].y, ...VEC});

  view.add(
    <>
      <Txt ref={title} y={-460} fontFamily={FONT} fontSize={46} fill={C.text} opacity={0} />
      <Node ref={grid}>
        {bands}
        {labels}
        {arrows.flat()}
        {chips}
      </Node>
      <Txt ref={wordLabel} x={colX(T - 1)} y={TOP_Y - 55} fontFamily={FONT} fontSize={24} fill={C.text} text="word vector" opacity={0} />
      <Txt ref={caption} y={495} fontFamily={FONT} fontSize={28} fill={C.dim} opacity={0} />
    </>,
  );
  grid().add([...emb.map(v => v.node), ...outs.flat().map(v => v.node)]);
  view.add(wordVec.node);

  function* embed(t: number, d: number): ThreadGenerator {
    const v = emb[t].node;
    v.y(CHIP_Y - 50);
    v.scale(0.3);
    yield* chips[t].scale(1.12, 0.15 * d).to(1, 0.15 * d);
    yield* all(v.opacity(1, 0.3 * d), v.y(EMB_Y, 0.6 * d, easeOutCubic), v.scale(1, 0.6 * d, easeOutCubic));
  }

  function* climb(t: number, d: number): ThreadGenerator {
    for (let l = 0; l < LEVELS.length; l++) {
      const v = outs[l][t];
      v.node.y(l === 0 ? EMB_Y : LEVELS[l - 1].y);
      yield* all(
        v.node.opacity(1, 0.2 * d),
        v.node.y(LEVELS[l].y, 0.5 * d, easeInOutCubic),
        v.set(outVals[l][t], 0.5 * d),
        t > 0 ? arrows[l][t - 1].end(1, 0.4 * d) : waitFor(0),
      );
    }
  }

  title().text('encoder: one word → one vector');
  yield* title().opacity(1, 0.5);
  yield* sequence(0.06, ...chips.slice(0, T - 1).map(c => c.opacity(1, 0.3)));
  yield* all(...bands.map(b => b.opacity(1, 0.5)), ...labels.map(l => l.opacity(1, 0.5)));

  // The first byte, slowly: lookup, then up through the three blocks.
  yield* waitUntil('lookup');
  caption().text('each byte looks up a learned vector');
  yield* caption().opacity(1, 0.4);
  yield* embed(0, 1.5);
  yield* waitFor(0.4);
  yield* caption().text('three blocks process it', 0.5);
  yield* climb(0, 1.5);

  // The rest of the word, faster: memory flows from letter to letter.
  yield* waitUntil('letters');
  yield* caption().text('memory carries from letter to letter', 0.5);
  for (let t = 1; t < T - 1; t++) {
    const d = Math.max(0.35, 1.1 - 0.25 * t);
    yield* embed(t, d);
    yield* climb(t, d);
  }

  // [W] closes the word.
  yield* waitUntil('end');
  yield* caption().text('[W]: the word is complete', 0.5);
  yield* chips[T - 1].opacity(1, 0.4);
  yield* embed(T - 1, 1);
  yield* climb(T - 1, 1);

  // Only the output at [W] is kept.
  yield* waitUntil('word_vector');
  const top = LEVELS.length - 1;
  wordVec.node.opacity(1);
  yield* all(
    caption().text('only the last output is kept: it has seen every letter', 0.6),
    ...outs[top].slice(0, T - 1).map(v => v.node.opacity(0.25, 0.6)),
    wordVec.node.y(TOP_Y, 0.8, easeInOutCubic),
    wordVec.node.scale(1.5, 0.8, easeInOutCubic),
    wordLabel().opacity(1, 0.6),
  );
  yield* waitFor(1);

  // Every word gets the same treatment: the backbone sees one vector per word.
  yield* waitUntil('all_words');
  const rowX = (i: number) => (i - (row.length - 1) / 2) * ROW_STEP;
  const rowChips = row.map(
    (w, i) =>
      new Rect({
        x: rowX(i),
        y: ROW_CHIP_Y,
        width: Math.max(64, [...glyph(w.text)].length * 20 + 30),
        height: 64,
        radius: 10,
        fill: C.panel,
        stroke: C.wire,
        lineWidth: 3,
        opacity: 0,
        children: [new Txt({fontFamily: FONT, fontSize: 30, fill: C.text, text: glyph(w.text)})],
      }),
  );
  // The first word's vector is the one just built.
  const rowVecs = row.slice(1).map((_, i) => new Vec(rowVals[i + 1], {x: rowX(i + 1), y: ROW_CHIP_Y, ...VEC}));
  view.add([...rowChips, ...rowVecs.map(v => v.node)]);

  yield* all(
    grid().opacity(0, 0.6),
    wordLabel().opacity(0, 0.4),
    caption().opacity(0, 0.4),
    title().text('one vector per word', 0.6),
  );
  yield* all(
    sequence(0.08, ...rowChips.map(c => c.opacity(1, 0.3))),
    wordVec.node.position([rowX(0), ROW_VEC_Y], 0.9, easeInOutCubic),
    wordVec.node.scale(1, 0.9, easeInOutCubic),
  );
  yield* sequence(
    0.12,
    ...rowVecs.map(v =>
      all(v.node.opacity(1, 0.3), v.node.y(ROW_VEC_Y, 0.6, easeOutCubic)),
    ),
  );
  caption().text('this sequence is what the backbone reads');
  yield* caption().opacity(1, 0.5);
  yield* waitFor(2);
});
