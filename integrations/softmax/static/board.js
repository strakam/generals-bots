// Tile rendering reused from generals-competition/board.js at 83a9b23.
// The adapter supplies visible snapshots; replay loading and simulation stay outside this renderer.
(() => {
  window.GeneralsTiles = function(container) {
    let tiles = [], els = [], selectedIndex = -1, arrows = new Map();
    function render() {
      for (let i = 0; i < tiles.length; i++) {
        const t = tiles[i];
        const el = els[i];
        let cls = 'tile';
        let content = '';
        // owner -2 = fog: neither player sees this cell. Army/ownership are
        // hidden, but mountains are static terrain and stay shown (grayer, with
        // the mountain icon) — like obstacles-in-fog in the original game.
        if (t.owner === -2) {
          cls += ' fog';
          if (t.mountain) { cls += ' fog-mountain has-mountain'; content = '<span class="icon"></span>'; }
        } else {
          if (t.mountain) cls += ' mountain has-mountain';
          else if (t.owner === 0) cls += ' blue';
          else if (t.owner === 1) cls += ' red';
          else if (t.castle && t.owner === -1) cls += ' neutral-castle';
          else cls += ' neutral';
          if (t.general) cls += ' has-general';
          else if (t.castle) cls += ' has-castle';

          if (t.general || t.castle) {
            content = '<span class="icon"></span>';
            if (t.count > 0) content += `<span class="num">${t.count}</span>`;
          } else if (t.mountain) {
            content = '<span class="icon"></span>';
          } else if (t.owner !== -1 && t.count > 0) {
            content = `<span class="num">${t.count}</span>`;
          }
        }
        if (i === selectedIndex) cls += ' selected';
        for (const [direction, pending] of arrows.get(i) || []) {
          content += `<span class="move-arrow ${pending ? 'queued' : 'submitted'}" data-direction="${direction}" aria-hidden="true"></span>`;
        }
        if (el.className !== cls) el.className = cls;
        if (el.innerHTML !== content) el.innerHTML = content;
      }
    }

    return {
      draw(frame, selected, queued = [], submitted = null) {
        const rows = frame.type_grid.length, cols = frame.type_grid[0].length;
        container.style.setProperty('--cols', cols);
        container.style.setProperty('--rows', rows);
        if (els.length !== rows * cols) {
          container.replaceChildren();
          els = Array.from({length: rows * cols}, () => {
            const el = document.createElement('div');
            el.className = 'tile'; container.appendChild(el); return el;
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
