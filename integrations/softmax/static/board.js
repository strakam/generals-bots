// Tile rendering reused from generals-competition/board.js at 83a9b23.
// The adapter supplies visible snapshots; replay loading and simulation stay outside this renderer.
(() => {
  window.GeneralsTiles = function(container) {
    let tiles = [], cells = [], selectedIndex = -1, arrows = new Map(), rows = 0, cols = 0;
    function render() {
      for (let i = 0; i < tiles.length; i++) {
        const t = tiles[i];
        const cell = cells[i], el = cell.el;
        let cls = 'tile';
        let icon = false, number = '';
        // owner -2 = fog: neither player sees this cell. Army/ownership are
        // hidden, but mountains are static terrain and stay shown (grayer, with
        // the mountain icon) — like obstacles-in-fog in the original game.
        if (t.owner === -2) {
          cls += ' fog';
          if (t.mountain) { cls += ' fog-mountain has-mountain'; icon = true; }
        } else {
          if (t.mountain) cls += ' mountain has-mountain';
          else if (t.owner === 0) cls += ' blue';
          else if (t.owner === 1) cls += ' red';
          else if (t.castle && t.owner === -1) cls += ' neutral-castle';
          else cls += ' neutral';
          if (t.general) cls += ' has-general';
          else if (t.castle) cls += ' has-castle';

          if (t.general || t.castle) {
            icon = true;
            if (t.count > 0) number = String(t.count);
          } else if (t.mountain) {
            icon = true;
          } else if (t.owner !== -1 && t.count > 0) {
            number = String(t.count);
          }
        }
        if (i === selectedIndex) cls += ' selected';
        if (el.className !== cls) el.className = cls;
        // Keep sprite masks and SVGs mounted while counts or queue state change.
        // Replacing innerHTML also reparses self-closing SVG paths, so comparing
        // that source with browser-serialized HTML rebuilt arrows on every draw.
        if (icon && !cell.icon) {
          cell.icon = document.createElement('span'); cell.icon.className = 'icon';
          el.prepend(cell.icon);
        } else if (!icon && cell.icon) { cell.icon.remove(); cell.icon = null; }
        if (number) {
          if (!cell.num) { cell.num = document.createElement('span'); cell.num.className = 'num'; el.append(cell.num); }
          if (cell.num.textContent !== number) cell.num.textContent = number;
        } else if (cell.num) { cell.num.remove(); cell.num = null; }
        const outgoing = arrows.get(i);
        for (const [direction, svg] of cell.arrows) {
          if (!outgoing?.has(direction)) { svg.remove(); cell.arrows.delete(direction); }
        }
        for (const [direction, pending] of outgoing || []) {
          let svg = cell.arrows.get(direction);
          if (!svg) {
            svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('data-direction', direction); svg.setAttribute('viewBox', '0 0 18 18');
            svg.setAttribute('aria-hidden', 'true');
            for (const cls of ['arrow-halo', 'arrow-body']) {
              const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
              path.setAttribute('class', cls); path.setAttribute('d', 'M3 6H9V3L15 9L9 15V12H3Z'); svg.append(path);
            }
            cell.arrows.set(direction, svg); el.append(svg);
          }
          const cls = `move-arrow ${pending ? 'queued' : 'submitted'}`;
          if (svg.getAttribute('class') !== cls) svg.setAttribute('class', cls);
        }
      }
    }

    return {
      draw(frame, selected, queued = [], submitted = null) {
        const nextRows = frame.type_grid.length, nextCols = frame.type_grid[0].length;
        if (cols !== nextCols) { cols = nextCols; container.style.setProperty('--cols', cols); }
        if (rows !== nextRows) { rows = nextRows; container.style.setProperty('--rows', rows); }
        if (cells.length !== rows * cols) {
          container.replaceChildren();
          cells = Array.from({length: rows * cols}, () => {
            const el = document.createElement('div');
            el.className = 'tile'; container.appendChild(el);
            return {el, icon: null, num: null, arrows: new Map()};
          });
        }
        selectedIndex = selected ? selected[0] * cols + selected[1] : -1;
        arrows = new Map();
        // One small arrow per outgoing edge, even when a route loops over it.
        for (const action of [submitted, ...queued]) {
          if (!action || action[0] !== 0) continue;
          const [, r, c, direction] = action, index = r * cols + c;
          if (!arrows.has(index)) arrows.set(index, new Map());
          arrows.get(index).set(direction, action !== submitted);
        }
        tiles = frame.type_grid.flatMap((row, r) => row.map((kind, c) => ({
          // Coworld uses 1=red and 2=blue; the competition renderer uses 0=blue and 1=red.
          owner: kind === 0 || kind === 5 ? -2 : frame.owner_grid[r][c] === 0 ? -1 :
            frame.owner_grid[r][c] === 1 ? 1 : 0,
          count: frame.army_grid[r][c], mountain: kind === 2 || kind === 5,
          general: kind === 4, castle: kind === 3,
        })));
        render();
      },
    };
  };
})();
