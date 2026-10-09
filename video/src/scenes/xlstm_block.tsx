import {Circle, Line, makeScene2D, Node, Rect, Txt} from '@motion-canvas/2d';
import {
  all,
  createRef,
  delay,
  easeInOutCubic,
  linear,
  Reference,
  sequence,
  waitFor,
  waitUntil,
} from '@motion-canvas/core';

import {C, FONT, follow, Grid, rmsNorm, rng, silu, Vec} from '../lib/viz';

// Horizontal positions of the pieces along the data path.
const IN = -870;
const SPLIT1 = -780;
const NORM1 = -690;
const CELL = -450;
const ADD1 = -210;
const SPLIT2 = -140;
const NORM2 = -60;
const FORK = 20;
const LIN = 110;
const WIDE_OUT = 200;
const SILU = 290;
const GATED = 380;
const MUL = 460;
const DOWN = 560;
const ADD2 = 680;
const OUT = 830;
// Vertical offset of the gate (up) and value (down) branches.
const BR = 130;
const BYPASS_Y = -340;
const CELL_SIZE = 280;

const W = 6;
// SwiGLU inner width is ~8/3 of the hidden width (`up_of` in config.rs).
const WIDE = 16;
const STEP = 22;
const WIDE_STEP = 12;

// Backbone layout from `src/model.rs`: every 4th block is an sLSTM.
const N_BLOCKS = 28;
const PICK = 5;
const isSlstm = (i: number) => i % 4 === 0;
const blockX = (i: number) => (i - (N_BLOCKS - 1) / 2) * 62;

// Illustrative values: the shapes are the real ones, the numbers are not.
// The residual stream runs at a larger scale than a norm's output, which is what
// makes each RMSNorm visibly dim its input.
const r = rng(11);
const rand = (n: number, scale: number) => Array.from({length: n}, () => (r() * 2 - 1) * scale);
const x = rand(W, 4.5);
const xn = rmsNorm(x);
const cellOut = rand(W, 2.2);
const z = x.map((v, i) => v + cellOut[i]);
const zn = rmsNorm(z);
const gate = rand(WIDE, 2.6);
const value = rand(WIDE, 2.2);
const gated = gate.map(silu);
const mixed = gated.map((g, i) => (g * value[i]) / 1.5);
const down = rand(W, 2.2);
const y = z.map((v, i) => v + down[i]);
const key = xn;
const val = xn.map((v, i) => xn[(i + 2) % W] - 0.5 * v);
const mem0 = Array.from({length: W}, () => rand(W, 2.2));
const memDecayed = mem0.map(row => row.map(v => v * 0.45));
const memWritten = memDecayed.map((row, i) => row.map((v, j) => v + val[i] * key[j] * 1.3));

function box(label: string, bx: number, by: number, width: number, height: number, labelY: number) {
  return (
    <>
      <Rect x={bx} y={by} width={width} height={height} radius={10} fill={C.panel} stroke={C.wire} lineWidth={2} />
      <Txt x={bx} y={labelY} fontFamily={FONT} fontSize={20} fill={C.dim} text={label} />
    </>
  );
}

function op(ref: Reference<Circle>, cx: number, symbol: string) {
  return (
    <Circle ref={ref} x={cx} size={50} fill={C.panel} stroke={C.text} lineWidth={3}>
      <Txt fontFamily={FONT} fontSize={30} fill={C.text} text={symbol} />
    </Circle>
  );
}

function wire(points: [number, number][], ref?: Reference<Line>) {
  return (
    <Line ref={ref} points={points} radius={22} stroke={C.wire} lineWidth={4} endArrow arrowSize={13} end={0} />
  );
}

export default makeScene2D(function* (view) {
  view.fill(C.bg);

  const title = createRef<Txt>();
  const stack = createRef<Node>();
  const token = createRef<Rect>();
  const diagram = createRef<Node>();
  const wires = createRef<Node>();
  const res1 = createRef<Line>();
  const res2 = createRef<Line>();
  const caption = createRef<Txt>();
  const add1 = createRef<Circle>();
  const add2 = createRef<Circle>();
  const mul = createRef<Circle>();
  const outLabel = createRef<Txt>();

  const blocks = Array.from({length: N_BLOCKS}, (_, i) => {
    const s = isSlstm(i);
    return new Rect({
      x: blockX(i),
      width: 50,
      height: 110,
      radius: 8,
      fill: C.panel,
      stroke: s ? C.blue : C.orange,
      lineWidth: 3,
      opacity: 0,
      children: [new Txt({fontFamily: FONT, fontSize: 22, fill: C.dim, text: s ? 's' : 'm'})],
    });
  });

  const half = CELL_SIZE / 2;
  view.add(
    <>
      <Txt ref={title} y={-460} fontFamily={FONT} fontSize={46} fill={C.text} opacity={0} />
      <Node ref={stack}>{blocks}</Node>
      <Rect ref={token} x={blockX(0) - 90} width={22} height={60} radius={4} fill={C.text} opacity={0} />
      <Node ref={diagram} y={40} scale={0.12} opacity={0}>
        <Node ref={wires}>
          {wire([[IN, 0], [CELL - half - 8, 0]])}
          {wire([[CELL + half, 0], [ADD1 - 30, 0]])}
          {wire([[ADD1, 0], [FORK, 0]])}
          {wire([[FORK, 0], [FORK, -BR], [MUL, -BR], [MUL, -30]])}
          {wire([[FORK, 0], [FORK, BR], [MUL, BR], [MUL, 30]])}
          {wire([[MUL, 0], [ADD2 - 30, 0]])}
          {wire([[ADD2, 0], [OUT + 70, 0]])}
          {wire([[SPLIT1, 0], [SPLIT1, BYPASS_Y], [ADD1, BYPASS_Y], [ADD1, -30]], res1)}
          {wire([[SPLIT2, 0], [SPLIT2, BYPASS_Y], [ADD2, BYPASS_Y], [ADD2, -30]], res2)}
        </Node>

        {box('RMSNorm', NORM1, 0, 50, 200, -122)}
        <Rect x={CELL} width={CELL_SIZE} height={CELL_SIZE} radius={14} fill={C.panel} stroke={C.orange} lineWidth={3} />
        <Txt x={CELL} y={-half + 26} fontFamily={FONT} fontSize={24} fill={C.orange} text="mLSTM" />
        <Txt x={CELL} y={half - 22} fontFamily={FONT} fontSize={18} fill={C.dim} text="memory" />
        <Txt ref={caption} x={CELL} y={half + 36} fontFamily={FONT} fontSize={22} fill={C.purple} opacity={0} />
        {op(add1, ADD1, '+')}
        {box('RMSNorm', NORM2, 0, 50, 200, -122)}

        <Rect x={(FORK + DOWN) / 2 + 20} width={DOWN - FORK + 60} height={570} radius={16} stroke={C.wire} lineWidth={2} lineDash={[8, 8]} />
        <Txt x={(FORK + DOWN) / 2 + 20} y={-305} fontFamily={FONT} fontSize={22} fill={C.dim} text="SwiGLU feed-forward" />
        {box('linear (gate)', LIN, -BR, 50, 220, -BR - 128)}
        {box('linear (value)', LIN, BR, 50, 220, BR + 128)}
        {box('SiLU', SILU, -BR, 50, 220, -BR - 128)}
        {op(mul, MUL, '⊙')}
        {box('linear (down)', DOWN, 0, 50, 200, -122)}
        {op(add2, ADD2, '+')}
        <Txt ref={outLabel} x={OUT} y={110} fontFamily={FONT} fontSize={20} fill={C.dim} text="to the next block" opacity={0} />
      </Node>
    </>,
  );

  // Vectors live inside the diagram so they share its coordinates and zoom.
  const wide = {step: WIDE_STEP, thickness: 18};
  const vx = new Vec(x, {x: IN - 60});
  const xCopy = new Vec(x, {x: SPLIT1});
  const mem = new Grid(mem0, {x: CELL, y: 8, step: 26});
  const vKey = new Vec(key, {x: CELL, y: -84, step: 26, thickness: 12, horizontal: true});
  const vVal = new Vec(val, {x: CELL - 104, y: 8, step: 26, thickness: 12});
  const vCell = new Vec(cellOut, {x: CELL + half - 20});
  const vz = new Vec(z, {x: ADD1});
  const zCopy = new Vec(z, {x: SPLIT2});
  const gIn = new Vec(zn, {x: FORK});
  const vIn = new Vec(zn, {x: FORK});
  const vGate = new Vec(gate, {x: LIN, y: -BR, ...wide});
  const vValue = new Vec(value, {x: LIN, y: BR, ...wide});
  const vMixed = new Vec(mixed, {x: MUL, ...wide});
  const vDown = new Vec(down, {x: DOWN});
  const vy = new Vec(y, {x: ADD2});
  xCopy.node.scale(0.55);
  zCopy.node.scale(0.55);
  diagram().add([mem.node, vKey.node, vVal.node, vx.node, xCopy.node, vCell.node, vz.node, zCopy.node]);
  diagram().add([gIn.node, vIn.node, vGate.node, vValue.node, vMixed.node, vDown.node, vy.node]);

  // The backbone: 28 blocks.
  title().text('backbone · 28 blocks');
  yield* title().opacity(1, 0.5);
  yield* sequence(0.025, ...blocks.map(b => b.opacity(1, 0.25)));
  yield* waitUntil('zoom');

  yield* blocks[PICK].fill(C.panelHi, 0.3);
  const zoom = 9;
  yield* all(
    title().text('one mLSTM block', 0.6),
    stack().scale(zoom, 1.2, easeInOutCubic),
    stack().x(-blockX(PICK) * zoom, 1.2, easeInOutCubic),
    stack().opacity(0, 1.0),
    diagram().scale(1, 1.2, easeInOutCubic),
    diagram().opacity(1, 0.8),
    mem.node.opacity(1, 0.8),
  );
  yield* all(...wires().children().map(w => (w as Line).end(1, 0.8)));

  // A word vector enters and is normalized.
  yield* waitUntil('input');
  yield* all(vx.node.opacity(1, 0.4), vx.node.x(SPLIT1, 0.6, easeInOutCubic));
  xCopy.node.opacity(0.9);
  yield* vx.node.x(NORM1, 0.5, easeInOutCubic);
  yield* vx.set(xn, 0.6);
  yield* vx.node.x(CELL - half - 30, 0.5, easeInOutCubic);

  // The memory stays in the cell: it fades a little, the word is written in, and the
  // output is read back out of it.
  yield* waitUntil('memory');
  yield* all(
    vx.node.opacity(0, 0.5),
    vx.node.x(CELL - half + 10, 0.5, easeInOutCubic),
    vKey.node.opacity(1, 0.5),
    vVal.node.opacity(1, 0.5),
  );
  caption().text('forget a little');
  yield* all(caption().opacity(1, 0.3), mem.set(memDecayed, 0.7));
  yield* waitFor(0.3);
  yield* caption().text('write the new word', 0.4);
  yield* sequence(0.1, ...memWritten.map((row, i) => mem.setRow(i, row, 0.35)));
  yield* waitFor(0.3);
  yield* all(caption().text('read out', 0.4), vKey.node.opacity(0, 0.4), vVal.node.opacity(0, 0.4));
  yield* all(
    vCell.node.opacity(1, 0.4),
    vCell.node.x(ADD1 - 70, 0.8, easeInOutCubic),
    delay(0.4, caption().text('kept for the next word', 0.4)),
  );

  // Residual: the input skips the cell and is added back.
  yield* waitUntil('residual');
  yield* all(
    res1().stroke(C.text, 0.3),
    follow(xCopy.node, res1(), 1.2),
    delay(0.8, caption().opacity(0, 0.4)),
  );
  yield* all(
    vCell.node.x(ADD1, 0.4),
    vCell.node.opacity(0, 0.4),
    xCopy.node.opacity(0, 0.3),
    add1().scale(1.3, 0.2).to(1, 0.2),
    vz.node.opacity(1, 0.4),
    res1().stroke(C.wire, 0.4),
  );
  yield* vz.node.x(SPLIT2, 0.4, easeInOutCubic);

  // Feed-forward: two linear layers from the same input, SiLU on the gate, multiply,
  // project back down, add the residual.
  yield* waitUntil('ffn');
  zCopy.node.opacity(0.9);
  yield* vz.node.x(NORM2, 0.4, easeInOutCubic);
  yield* vz.set(zn, 0.6);
  yield* vz.node.x(FORK, 0.4, easeInOutCubic);
  gIn.node.opacity(1);
  vIn.node.opacity(1);
  vz.node.opacity(0);
  yield* all(
    gIn.node.position([LIN - 60, -BR], 0.6, easeInOutCubic),
    vIn.node.position([LIN - 60, BR], 0.6, easeInOutCubic),
  );
  // The small vector goes into the layer first; only then does the wide one come out.
  yield* all(
    gIn.node.x(LIN, 0.4, easeInOutCubic),
    vIn.node.x(LIN, 0.4, easeInOutCubic),
    gIn.node.scale.x(0, 0.4),
    vIn.node.scale.x(0, 0.4),
    gIn.node.opacity(0, 0.4),
    vIn.node.opacity(0, 0.4),
  );
  const toWide = (W * STEP) / (WIDE * WIDE_STEP);
  vGate.node.scale([0, toWide]);
  vValue.node.scale([0, toWide]);
  yield* all(
    vGate.node.opacity(1, 0.2),
    vValue.node.opacity(1, 0.2),
    vGate.node.scale([1, 1], 0.6, easeInOutCubic),
    vValue.node.scale([1, 1], 0.6, easeInOutCubic),
    vGate.node.x(WIDE_OUT, 0.6, easeInOutCubic),
    vValue.node.x(WIDE_OUT, 0.6, easeInOutCubic),
  );
  yield* vGate.node.x(SILU, 0.5, easeInOutCubic);
  yield* vGate.set(gated, 0.6);
  yield* all(vGate.node.x(GATED, 0.5, easeInOutCubic), vValue.node.x(GATED, 0.5, easeInOutCubic));
  yield* all(
    vGate.node.position([MUL, -40], 0.6, easeInOutCubic),
    vValue.node.position([MUL, 40], 0.6, easeInOutCubic),
    vGate.node.opacity(0, 0.6),
    vValue.node.opacity(0, 0.6),
    mul().scale(1.3, 0.3).to(1, 0.3),
  );
  yield* all(vMixed.node.opacity(1, 0.3), vMixed.node.x(DOWN - 60, 0.6, easeInOutCubic));
  yield* all(
    vMixed.node.x(DOWN, 0.4, easeInOutCubic),
    vMixed.node.scale.x(0, 0.4),
    vMixed.node.opacity(0, 0.4),
  );
  vDown.node.scale([0, 1 / toWide]);
  yield* all(
    vDown.node.opacity(1, 0.2),
    vDown.node.scale([1, 1], 0.6, easeInOutCubic),
    vDown.node.x(ADD2 - 70, 0.8, easeInOutCubic),
  );
  yield* all(res2().stroke(C.text, 0.3), follow(zCopy.node, res2(), 1.4));
  yield* all(
    vDown.node.x(ADD2, 0.4),
    vDown.node.opacity(0, 0.4),
    zCopy.node.opacity(0, 0.3),
    add2().scale(1.3, 0.2).to(1, 0.2),
    vy.node.opacity(1, 0.4),
    res2().stroke(C.wire, 0.4),
  );
  yield* all(vy.node.x(OUT, 0.6, easeInOutCubic), outLabel().opacity(1, 0.5));
  yield* waitFor(0.6);

  // Back out: the vector runs through the whole stack.
  yield* waitUntil('stack');
  yield* all(
    title().text('× 28', 0.6),
    diagram().scale(0.12, 1.2, easeInOutCubic),
    diagram().opacity(0, 1.0),
    stack().scale(1, 1.2, easeInOutCubic),
    stack().x(0, 1.2, easeInOutCubic),
    stack().opacity(1, 0.8),
    blocks[PICK].fill(C.panel, 1.2),
  );
  yield* token().opacity(1, 0.3);
  yield* all(
    token().x(blockX(N_BLOCKS - 1) + 90, 2.8, linear),
    sequence(2.8 / N_BLOCKS, ...blocks.map(b => b.fill(C.panelHi, 0.08).to(C.panel, 0.4))),
  );
  yield* token().opacity(0, 0.3);
  yield* waitFor(1);
});
