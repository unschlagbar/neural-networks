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
import {C, FONT, follow, rng, Vec} from '../lib/viz';

// The three stages of `src/model.rs` as one recurrent loop: a word is encoded, the
// backbone folds it into its memory, the decoder spells the next word, and that word
// goes back in as the next input.
const OUT_Y = -390;
const DEC_Y = -240;
const CTX_Y = -120;
const BB_Y = 0;
const WV_Y = 120;
const ENC_Y = 240;
const IN_Y = 390;
const LOOP_X = 520;
const TEXT_Y = -470;
const TEXT_X0 = -560;
const S = 30;
const ITERATIONS = 4;

const ENC = C.green;
const BB = C.orange;
const DEC = C.blue;

const glyphs = (s: string) => [...s].map(c => (c === ' ' ? '·' : c === '\n' ? '⏎' : c));
const words = data.words.slice(0, ITERATIONS + 1);
const CHAR_W = 0.6 * 30;

// Illustrative values: the shapes are the real ones, the numbers are not.
const r = rng(41);
const rand = () => Array.from({length: 6}, () => (r() * 2 - 1) * 2.8);
const memVals = [rand(), ...words.map(rand)];
const wordVals = words.map(rand);
const ctxVals = words.map(rand);

function stage(y: number, width: number, height: number, name: string, sub: string, color: string) {
  return new Rect({
    y,
    width,
    height,
    radius: 14,
    fill: C.panel,
    stroke: color,
    lineWidth: 3,
    opacity: 0,
    children: [
      new Txt({y: -height / 2 + 30, fontFamily: FONT, fontSize: 26, fill: color, text: name}),
      new Txt({y: -height / 2 + 58, fontFamily: FONT, fontSize: 16, fill: C.dim, text: sub}),
    ],
  });
}

function chip(text: string, y: number) {
  const g = glyphs(text).join('');
  return new Rect({
    y,
    width: Math.max(64, g.length * 20 + 30),
    height: 60,
    radius: 10,
    fill: C.panel,
    stroke: C.wire,
    lineWidth: 3,
    opacity: 0,
    children: [new Txt({fontFamily: FONT, fontSize: 30, fill: C.text, text: g})],
  });
}

export default makeScene2D(function* (view) {
  view.fill(C.bg);

  const title = createRef<Txt>();
  const caption = createRef<Txt>();
  const root = createRef<Node>();
  const memLoop = createRef<Line>();
  const wordLoop = createRef<Line>();
  const memLabel = createRef<Txt>();

  const encoder = stage(ENC_Y, 320, 100, 'encoder', '256 wide · fresh for every word', ENC);
  const backbone = stage(BB_Y, 400, 150, 'backbone', '1024 wide · 28 blocks', BB);
  const decoder = stage(DEC_Y, 320, 100, 'decoder', '256 wide · fresh for every word', DEC);
  const memory = new Vec(memVals[0], {y: BB_Y + 40, step: 26, thickness: 20, horizontal: true});

  // Text so far: the prompt, then every generated word appended.
  const textPieces: Txt[] = [];
  let textX = TEXT_X0;
  for (const w of words) {
    const g = glyphs(w.text).join('');
    textPieces.push(
      new Txt({x: textX, y: TEXT_Y, offset: [-1, 0], fontFamily: FONT, fontSize: 30, fill: C.text, text: g, opacity: 0}),
    );
    textX += g.length * CHAR_W + 14;
  }

  view.add(
    <>
      <Txt ref={title} y={TEXT_Y} fontFamily={FONT} fontSize={46} fill={C.text} opacity={0} />
      <Node ref={root}>
        <Line
          ref={memLoop}
          points={[[200, BB_Y - 40], [270, BB_Y - 40], [270, BB_Y + 40], [200, BB_Y + 40]]}
          radius={24}
          stroke={BB}
          lineWidth={4}
          endArrow
          arrowSize={12}
          end={0}
        />
        <Txt ref={memLabel} x={340} y={BB_Y} fontFamily={FONT} fontSize={20} fill={BB} text="memory" opacity={0} />
        <Line
          ref={wordLoop}
          points={[[170, OUT_Y], [LOOP_X, OUT_Y], [LOOP_X, IN_Y], [110, IN_Y]]}
          radius={40}
          stroke={C.wire}
          lineWidth={4}
          lineDash={[12, 10]}
          endArrow
          arrowSize={14}
          end={0}
        />
      </Node>
      <Txt ref={caption} y={485} fontFamily={FONT} fontSize={28} fill={C.dim} opacity={0} />
    </>,
  );
  root().add([encoder, backbone, decoder, memory.node, ...textPieces]);

  const pulse = (node: Rect) => node.scale(1.06, 0.15).to(1, 0.25);

  /** One word through the loop: encode word k, update memory, spell word k + 1. */
  function* iteration(k: number, d: number, captions: boolean): ThreadGenerator {
    const input = k === 0 ? firstChip : incoming!;
    const wv = new Vec(wordVals[k], {y: ENC_Y, step: 18, thickness: 18, horizontal: true});
    const ctx = new Vec(ctxVals[k], {y: BB_Y, step: 18, thickness: 18, horizontal: true});
    const next = [...glyphs(words[k + 1].text), 'W'];
    const cells = next.map(
      (ch, c) =>
        new Rect({
          x: (c - (next.length - 1) / 2) * S,
          y: OUT_Y,
          width: S - 4,
          height: S + 10,
          radius: 4,
          fill: C.panel,
          stroke: c === next.length - 1 ? C.purple : DEC,
          lineWidth: 2,
          opacity: 0,
          children: [
            new Txt({
              fontFamily: FONT,
              fontSize: c === next.length - 1 ? 13 : 20,
              fill: c === next.length - 1 ? C.purple : C.text,
              text: ch,
            }),
          ],
        }),
    );
    root().add([wv.node, ctx.node, ...cells]);

    if (captions) yield* waitUntil('encoder');
    if (captions) yield* caption().text('encoder: turns the word into one vector', 0.5);
    yield* all(input.y(ENC_Y, 0.6 * d, easeInOutCubic), input.opacity(0, 0.6 * d), input.scale(0.6, 0.6 * d));
    yield* pulse(encoder);
    yield* all(wv.node.opacity(1, 0.3 * d), wv.node.y(WV_Y, 0.5 * d, easeOutCubic));

    if (captions) yield* waitUntil('backbone');
    if (captions) yield* caption().text('backbone: folds it into its memory of the text so far', 0.5);
    yield* all(wv.node.y(BB_Y + 40, 0.5 * d, easeInOutCubic), wv.node.opacity(0, 0.5 * d));
    yield* all(
      pulse(backbone),
      memory.set(memVals[k + 1], 0.6 * d),
      memLoop().stroke(C.text, 0.2 * d).to(BB, 0.5 * d),
    );
    yield* all(ctx.node.opacity(1, 0.3 * d), ctx.node.y(CTX_Y, 0.5 * d, easeOutCubic));

    if (captions) yield* waitUntil('decoder');
    if (captions) yield* caption().text('decoder: spells out the next word, letter by letter', 0.5);
    yield* all(ctx.node.y(DEC_Y, 0.5 * d, easeInOutCubic), ctx.node.opacity(0, 0.5 * d));
    yield* pulse(decoder);
    yield* sequence(
      0.2 * d,
      ...cells.map(c => all(c.opacity(1, 0.2 * d), c.y(OUT_Y - 10, 0).to(OUT_Y, 0.3 * d, easeOutCubic))),
    );

    if (captions) yield* waitUntil('loop');
    if (captions) yield* caption().text('the new word goes back in as the next input', 0.5);
    incoming = chip(words[k + 1].text, OUT_Y);
    root().add(incoming);
    yield* all(
      ...cells.map(c => all(c.x(0, 0.4 * d, easeInOutCubic), c.opacity(0, 0.4 * d))),
      incoming.opacity(1, 0.4 * d),
      textPieces[k + 1].opacity(1, 0.4 * d),
    );
    yield* all(incoming.x(170, 0.3 * d, easeInOutCubic), captions ? wordLoop().end(1, 0.6) : waitFor(0));
    yield* follow(incoming, wordLoop(), 1.2 * d);
    yield* incoming.x(0, 0.3 * d, easeInOutCubic);
    cells.forEach(c => c.remove());
    wv.node.remove();
    ctx.node.remove();
  }

  const firstChip = chip(words[0].text, IN_Y);
  let incoming: Rect | null = null;
  root().add(firstChip);

  // The stack, built bottom-up.
  title().text('three stages, one loop');
  yield* title().opacity(1, 0.5);
  yield* sequence(0.2, encoder.opacity(1, 0.5), backbone.opacity(1, 0.5), decoder.opacity(1, 0.5));
  yield* all(memory.node.opacity(1, 0.5), memLoop().end(1, 0.6), memLabel().opacity(1, 0.5));
  yield* waitFor(0.4);

  caption().text('the prompt, split into words');
  yield* all(title().opacity(0, 0.4), caption().opacity(1, 0.5), firstChip.opacity(1, 0.5), textPieces[0].opacity(1, 0.5));

  // First word slowly, with a marker per stage.
  yield* iteration(0, 1, true);

  // Then word after word.
  yield* waitUntil('repeat');
  yield* caption().text('and again, word by word: only the backbone keeps memory', 0.5);
  for (let k = 1; k < ITERATIONS; k++) {
    yield* iteration(k, 0.5, false);
  }
  yield* waitFor(0.5);

  // Into the encoder.
  yield* waitUntil('zoom_encoder');
  const zoom = 5;
  yield* all(
    caption().opacity(0, 0.5),
    root().scale(zoom, 1.2, easeInOutCubic),
    root().y(-ENC_Y * zoom, 1.2, easeInOutCubic),
    root().opacity(0, 1.2),
  );
});
