import {Line, Node, Rect} from '@motion-canvas/2d';
import {all, easeInOutCubic, ThreadGenerator, TimingFunction, tween} from '@motion-canvas/core';

export const FONT = 'JetBrains Mono, DejaVu Sans Mono, monospace';

export const C = {
  bg: '#11131a',
  panel: '#1e2230',
  panelHi: '#2c3350',
  cellBase: '#2a2f40',
  wire: '#454c66',
  text: '#e8e8f0',
  dim: '#8a90a8',
  orange: '#e0b35a',
  blue: '#6fb3e0',
  purple: '#b58be0',
  green: '#6fd49a',
};

/** Deterministic PRNG (mulberry32), so a scene renders the same every time. */
export function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const silu = (x: number) => x / (1 + Math.exp(-x));

export function rmsNorm(a: number[]) {
  const s = Math.sqrt(a.reduce((acc, v) => acc + v * v, 0) / a.length) || 1;
  return a.map(v => v / s);
}

const hex = (h: string) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));

/** |v| at which a cell reaches full colour. Linear below it, so scale changes stay visible. */
export const FULL_SCALE = 3;

/** Diverging colour: negative blue, positive orange, zero the neutral cell colour. */
export function valueColor(v: number) {
  const base = hex(C.cellBase);
  const target = hex(v < 0 ? C.blue : C.orange);
  const t = Math.min(1, Math.abs(v) / FULL_SCALE);
  const [r, g, b] = base.map((c, i) => Math.round(c + (target[i] - c) * t));
  return `rgb(${r},${g},${b})`;
}

/** A vector drawn as a column (or row) of coloured cells. */
export class Vec {
  readonly node: Node;
  readonly cells: Rect[];

  constructor(
    values: number[],
    opts: {x?: number; y?: number; step?: number; thickness?: number; horizontal?: boolean} = {},
  ) {
    const step = opts.step ?? 22;
    const thickness = opts.thickness ?? 22;
    const along = (i: number) => (i - (values.length - 1) / 2) * step;
    this.node = new Node({x: opts.x ?? 0, y: opts.y ?? 0, opacity: 0});
    this.cells = values.map(
      (v, i) =>
        new Rect({
          x: opts.horizontal ? along(i) : 0,
          y: opts.horizontal ? 0 : along(i),
          width: opts.horizontal ? step - 3 : thickness,
          height: opts.horizontal ? thickness : step - 3,
          radius: 3,
          fill: valueColor(v),
        }),
    );
    this.node.add(this.cells);
  }

  set(values: number[], duration = 0.5): ThreadGenerator {
    return all(...this.cells.map((c, i) => c.fill(valueColor(values[i]), duration)));
  }
}

/** A matrix drawn as a grid of coloured cells. */
export class Grid {
  readonly node: Node;
  readonly cells: Rect[][];

  constructor(values: number[][], opts: {x?: number; y?: number; step?: number} = {}) {
    const step = opts.step ?? 26;
    const rows = values.length;
    const cols = values[0].length;
    this.node = new Node({x: opts.x ?? 0, y: opts.y ?? 0, opacity: 0});
    this.cells = values.map((row, r) =>
      row.map(
        (v, c) =>
          new Rect({
            x: (c - (cols - 1) / 2) * step,
            y: (r - (rows - 1) / 2) * step,
            width: step - 3,
            height: step - 3,
            radius: 3,
            fill: valueColor(v),
          }),
      ),
    );
    this.node.add(this.cells.flat());
  }

  set(values: number[][], duration = 0.5): ThreadGenerator {
    return all(
      ...this.cells.flatMap((row, r) => row.map((cell, c) => cell.fill(valueColor(values[r][c]), duration))),
    );
  }

  setRow(r: number, values: number[], duration = 0.4): ThreadGenerator {
    return all(...this.cells[r].map((cell, c) => cell.fill(valueColor(values[c]), duration)));
  }
}

/** Moves `node` along `line`; both must share a parent. */
export function follow(
  node: Node,
  line: Line,
  duration: number,
  ease: TimingFunction = easeInOutCubic,
): ThreadGenerator {
  return tween(duration, t => node.position(line.getPointAtPercentage(ease(t)).position));
}
