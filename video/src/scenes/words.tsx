import {makeScene2D, Node, Rect, Txt} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  waitFor,
} from '@motion-canvas/core';

import data from '../data/words.json';

const CELL = 46;
const WORD_GAP = 22;
const ROW = 120;
const FONT = 'JetBrains Mono, DejaVu Sans Mono, monospace';
const PALETTE = ['#e0b35a', '#6fb3e0', '#b58be0', '#6fd49a', '#e07a6f', '#d4d46f'];

const glyph = (c: string) => (c === ' ' ? '·' : c === '\n' ? '⏎' : c === '\t' ? '⇥' : c);

interface Cell {
  ch: string;
  byte: number;
  word: number;
  tightX: number;
  splitX: number;
  y: number;
}

interface Word {
  x: number;
  y: number;
  width: number;
  color: string;
}

// Lines break after every word that carries a newline, like the source code does.
function layout() {
  const cells: Cell[] = [];
  const words: Word[] = [];
  const lines: number[][] = [[]];
  data.words.forEach((w, i) => {
    lines[lines.length - 1].push(i);
    if (w.text.includes('\n')) lines.push([]);
  });
  if (lines[lines.length - 1].length === 0) lines.pop();

  lines.forEach((line, li) => {
    const y = (li - (lines.length - 1) / 2) * ROW;
    const chars = line.reduce((n, wi) => n + [...data.words[wi].text].length, 0);
    const tightStart = -((chars - 1) * CELL) / 2;
    const splitStart = -((chars - 1) * CELL + (line.length - 1) * WORD_GAP) / 2;
    let k = 0;
    line.forEach((wi, j) => {
      const text = [...data.words[wi].text];
      const bytes = data.words[wi].bytes;
      const first = splitStart + k * CELL + j * WORD_GAP;
      text.forEach((ch, c) => {
        cells.push({
          ch,
          byte: bytes[c] ?? 0,
          word: wi,
          tightX: tightStart + k * CELL,
          splitX: splitStart + k * CELL + j * WORD_GAP,
          y,
        });
        k++;
      });
      words.push({
        x: first + ((text.length - 1) * CELL) / 2,
        y,
        width: text.length * CELL + 10,
        color: PALETTE[wi % PALETTE.length],
      });
    });
  });
  return {cells, words};
}

export default makeScene2D(function* (view) {
  view.fill('#11131a');
  const {cells, words} = layout();
  const nBytes = data.words.reduce((n, w) => n + w.bytes.length, 0);

  const title = createRef<Txt>();
  const caption = createRef<Txt>();
  const cellRefs = cells.map(() => createRef<Rect>());
  const byteRefs = cells.map(() => createRef<Txt>());
  const wordRefs = words.map(() => createRef<Rect>());

  view.add(
    <>
      <Txt ref={title} y={-440} fontFamily={FONT} fontSize={48} fill="#e8e8f0" opacity={0}>
        text → bytes → words
      </Txt>
      <Node y={20}>
        {words.map((w, i) => (
          <Rect
            ref={wordRefs[i]}
            x={w.x}
            y={w.y}
            width={w.width}
            height={92}
            radius={12}
            fill={w.color + '26'}
            stroke={w.color}
            lineWidth={3}
            opacity={0}
          />
        ))}
        {cells.map((c, i) => (
          <Rect
            ref={cellRefs[i]}
            x={c.tightX}
            y={c.y + 30}
            width={CELL - 6}
            height={80}
            radius={6}
            fill="#1e2230"
            opacity={0}
          >
            <Txt y={-10} fontFamily={FONT} fontSize={30} fill="#e8e8f0" text={glyph(c.ch)} />
            <Txt
              ref={byteRefs[i]}
              y={24}
              fontFamily={FONT}
              fontSize={15}
              fill="#8a90a8"
              text={String(c.byte)}
              opacity={0}
            />
          </Rect>
        ))}
      </Node>
      <Txt ref={caption} y={440} fontFamily={FONT} fontSize={36} fill="#8a90a8" opacity={0} />
    </>,
  );

  yield* title().opacity(1, 0.6);

  // Plain text: one cell per character.
  yield* sequence(
    0.015,
    ...cells.map((c, i) =>
      all(cellRefs[i]().opacity(1, 0.3), cellRefs[i]().y(c.y, 0.4, easeOutCubic)),
    ),
  );
  yield* waitFor(0.6);

  // Every character is really one or more UTF-8 bytes.
  caption().text(`${nBytes} bytes`);
  yield* all(caption().opacity(1, 0.5), ...byteRefs.map(r => r().opacity(1, 0.6)));
  yield* waitFor(1);

  // The segmenter groups bytes into words.
  yield* all(...cells.map((c, i) => cellRefs[i]().x(c.splitX, 0.9, easeInOutCubic)));
  yield* sequence(0.05, ...wordRefs.map(r => r().opacity(1, 0.35)));
  yield* caption().text(`${nBytes} bytes → ${data.words.length} words`, 0.6);

  yield* waitFor(2);
});
