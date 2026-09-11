/* Coworld transport and controls around the existing competition tile renderer. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const boardElement = $('board'), renderer = GeneralsTiles(boardElement);
  const params = new URLSearchParams(location.search);
  const replayURL = new URLSearchParams(location.hash.slice(1)).get('replay') || params.get('replay');
  const livePrefix = location.pathname.includes('/client/') ? location.pathname.split('/client/')[0] : '';
  let board = null, slot = null, selected = null, half = false, ws = null;
  let currentTurn = -1, sent = true, passTimer = null, replay = null, frameIndex = 0;
  let playing = true, replayTimer = null, readySent = false, replayLoad = 0;
  const moves = [[-1,0],[1,0],[0,-1],[0,1]], queue = [];
  let inFlight = null;
  const sameTile = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];
  const canQueue = () => slot !== null && board && ws?.readyState === WebSocket.OPEN && !replay;
  const post = data => { if (parent !== window) parent.postMessage({src: 'coworld-replay', ...data}, '*'); };
  function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
  function ready() { if (!readySent) { readySent = true; setTimeout(() => post({type: 'ready'}), 0); } }
  function fail(message) {
    status(message, true); $('cover').hidden = false;
    $('cover-title').textContent = 'Unable to open this match'; $('cover-text').textContent = message;
    post({type: 'error', message});
  }
  function names(players) { players.forEach((name, i) => { $(`name${i}`).textContent = name; }); }
  function scoreboard(turn, army, land) {
    $('turn').textContent = turn;
    for (let i = 0; i < 2; i++) { $(`army${i}`).textContent = army[i]; $(`land${i}`).textContent = land[i]; }
    $('endgame').textContent = turn >= 800 ? 'DEATHTOUCH ACTIVE · Reach the enemy general to win' : 'Deathtouch from turn 800';
  }
  function draw() {
    if (!board) return;
    renderer.draw(board, selected, queue.map(item => item.action), inFlight?.action);
    $('queue-count').textContent = `${queue.length} queued`;
    $('cover').hidden = true; ready();
  }
  function send(action) {
    if (!ws || ws.readyState !== WebSocket.OPEN || sent || currentTurn < 0) return false;
    ws.send(JSON.stringify({type: 'action', turn: currentTurn, action}));
    sent = true; clearTimeout(passTimer);
    status(action[0] === 1 ? 'Holding position.' : action[0] === 2 ? 'Castle requested.' : 'Move submitted.');
    return true;
  }
  function dispatchQueue() {
    if (!canQueue() || sent || !queue.length) return;
    const [kind, r, c, direction] = queue[0].action;
    if (kind !== 1 && board.owner_grid[r][c] !== slot+1) {
      clearQueue();
      status('Queue stopped: the next source tile is no longer yours.'); return;
    }
    if (kind === 0) {
      const [dr, dc] = moves[direction];
      if (board.type_grid[r+dr][c+dc] === 2) {
        clearQueue(); status('Queue stopped at a mountain.'); return;
      }
      if (board.army_grid[r][c] < 2) {
        status('Waiting for an army to move · E undoes one, Q clears the queue.'); return;
      }
    }
    if (send(queue[0].action)) inFlight = queue.shift();
    draw();
  }
  function enqueue(action, to = selected) {
    if (!canQueue()) return;
    queue.push({action, from: selected, to}); selected = to;
    status(`${queue.length} queued · E undoes one, Q clears the queue.`);
    dispatchQueue(); draw();
  }
  function undoMove() {
    const removed = queue.pop();
    if (removed && sameTile(selected, removed.to)) selected = removed.from;
    draw(); status(removed ? 'Last queued action removed.' : 'No queued moves to undo.');
  }
  function clearQueue() {
    if (queue.length && sameTile(selected, queue[queue.length-1].to)) selected = queue[0].from;
    queue.length = 0; draw();
    status(inFlight ? 'Queue cleared. The submitted action will finish this turn.' : 'Queue cleared.');
  }
  function move(direction) {
    if (!selected || !canQueue()) return;
    const [r, c] = selected, delta = moves[direction];
    const nr = r+delta[0], nc = c+delta[1];
    if (nr < 0 || nc < 0 || nr >= board.type_grid.length || nc >= board.type_grid[0].length) return;
    if (board.type_grid[nr][nc] === 2) return;
    enqueue([0,r,c,direction,half ? 1 : 0], [nr,nc]);
  }
  boardElement.addEventListener('click', event => {
    if (!canQueue()) return;
    const rect = boardElement.getBoundingClientRect();
    const r = Math.floor((event.clientY-rect.top)/rect.height*board.type_grid.length);
    const c = Math.floor((event.clientX-rect.left)/rect.width*board.type_grid[0].length);
    if (r < 0 || c < 0 || r >= board.type_grid.length || c >= board.type_grid[0].length) return;
    if (selected) {
      const dr = r-selected[0], dc = c-selected[1];
      if (Math.abs(dr)+Math.abs(dc) === 1) { move(dr < 0 ? 0 : dr > 0 ? 1 : dc < 0 ? 2 : 3); boardElement.focus(); return; }
    }
    if (board.owner_grid[r][c] === slot+1) selected = [r,c];
    draw(); boardElement.focus();
  });
  $('split').onclick = () => { half = !half; $('split').setAttribute('aria-pressed', String(half)); };
  $('build').onclick = () => { if (selected) enqueue([2, ...selected, 0, 0]); };
  $('pass').onclick = () => enqueue([1,0,0,0,0]);
  $('undo').onclick = undoMove;
  $('clear').onclick = clearQueue;
  // Keep keyboard play working after clicking a control with the mouse.
  $('play-controls').addEventListener('click', event => { if (event.target.closest('button')) boardElement.focus(); });
  document.addEventListener('keydown', event => {
    if (slot === null || event.ctrlKey || event.metaKey || event.altKey ||
        event.target.matches('input, textarea, select, button') || event.target.isContentEditable) return;
    const key = event.key.toLowerCase();
    const keys = {arrowup:0, w:0, arrowdown:1, s:1, arrowleft:2, a:2, arrowright:3, d:3};
    if (key in keys) { event.preventDefault(); move(keys[key]); }
    else if (key === 'h') $('split').click();
    else if (key === 'b') $('build').click();
    else if (key === 'e') { event.preventDefault(); undoMove(); }
    else if (key === 'q') { event.preventDefault(); clearQueue(); }
    else if (key === ' ') { event.preventDefault(); $('pass').click(); }
  });
  function outcome(result) {
    const text = result.winner < 0 ? 'Draw' : `${$('name'+result.winner).textContent} wins`;
    const reasons = {general_capture:'general captured', turn_limit:'turn limit reached', forfeit:'opponent timed out', double_forfeit:'both players timed out'};
    return `${text} · ${reasons[result.reason] || result.reason}`;
  }
  function showReplayFrame() {
    board = replay.frames[frameIndex]; scoreboard(board.turn, board.army, board.land); draw();
    $('seek').value = frameIndex;
    status(`${outcome(replay.result)} · Frame ${frameIndex+1} of ${replay.frames.length}`);
  }
  function resetReplayTimer() {
    clearInterval(replayTimer);
    replayTimer = setInterval(() => {
      if (playing && replay) { frameIndex = (frameIndex+1) % replay.frames.length; showReplayFrame(); }
    }, 1000 / Number($('speed').value));
  }
  $('pause').onclick = () => { playing = !playing; $('pause').textContent = playing ? 'Pause' : 'Play'; };
  $('restart').onclick = () => { frameIndex = 0; showReplayFrame(); };
  $('seek').oninput = () => { frameIndex = Number($('seek').value); showReplayFrame(); };
  $('speed').onchange = resetReplayTimer;
  async function loadReplay(url) {
    const load = ++replayLoad;
    clearInterval(replayTimer); clearTimeout(passTimer);
    replay = null; board = null; readySent = false; slot = null; sent = true;
    queue.length = 0; inFlight = null; selected = null;
    $('play-controls').hidden = true;
    $('mode').textContent = 'LOADING'; $('replay-controls').hidden = true;
    $('cover').hidden = false; $('cover-title').textContent = 'Loading the replay';
    $('cover-text').textContent = 'Reconstructing the battlefield.';
    post({type:'loading'}); post({type:'phase', phase:'replay_fetch_start'});
    const response = await fetch(url, {credentials:'omit'});
    if (!response.ok) throw new Error(`Replay could not be loaded (${response.status}).`);
    let bytes = new Uint8Array(await response.arrayBuffer());
    const compressed = bytes[0] === 0x1f && bytes[1] === 0x8b;
    post({type:'phase', phase:'replay_fetch_end', bytes:bytes.length, compressed});
    if (compressed) bytes = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    const data = JSON.parse(new TextDecoder().decode(bytes));
    if (load !== replayLoad) return;
    if (data.format !== 'generals-coworld' || data.version !== 1 || !Array.isArray(data.frames) || !data.frames.length || data.frames.length > 1201 ||
        !Number.isInteger(data.height) || !Number.isInteger(data.width) || data.height < 1 || data.height > 21 || data.width < 1 || data.width > 21 ||
        !Array.isArray(data.players) || data.players.length !== 2 || !data.result) throw new Error('Unsupported or incomplete Generals replay.');
    for (const frame of data.frames) for (const key of ['type_grid','owner_grid','army_grid']) {
      if (!Array.isArray(frame[key]) || frame[key].length !== data.height || frame[key].some(row => !Array.isArray(row) || row.length !== data.width || row.some(n => !Number.isInteger(n)))) throw new Error('Replay contains an invalid board.');
    }
    replay = data; slot = null; selected = null; clearTimeout(passTimer);
    names(data.players); $('mode').textContent = 'REPLAY';
    $('play-controls').hidden = true; $('replay-controls').hidden = false; $('seek').max = data.frames.length-1;
    post({type:'phase', phase:'replay_parsed'});
    readySent = false; frameIndex = 0; showReplayFrame(); resetReplayTimer();
  }
  function observe(message) {
    slot = message.slot; currentTurn = message.turn; sent = false; inFlight = null; names(message.players);
    const owners = message.owner_grid.map(row => row.map(o => o === 0 ? 0 : o === 1 ? slot+1 : 2-slot));
    board = {...message, owner_grid: owners};
    if (!queue.length && selected && owners[selected[0]][selected[1]] !== slot+1) selected = null;
    if (!selected) {
      for (let r=0; r<owners.length; r++) for (let c=0; c<owners[0].length; c++) {
        if (owners[r][c] === slot+1 && board.type_grid[r][c] === 4) selected = [r,c];
      }
    }
    const army = [], land = [];
    army[slot] = message.my_army; army[1-slot] = message.opp_army;
    land[slot] = message.my_land; land[1-slot] = message.opp_land;
    scoreboard(message.turn, army, land); draw();
    status('Your move · Select a tile and a direction.');
    clearTimeout(passTimer);
    passTimer = setTimeout(() => send([1,0,0,0,0]), message.turn_timeout_seconds * 800);
    dispatchQueue();
  }
  function connectLive() {
    const isPlayer = location.pathname.endsWith('/client/player');
    const route = livePrefix + (isPlayer ? '/player' : '/global');
    const url = new URL(route, location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    if (isPlayer) { url.searchParams.set('slot', params.get('slot') || ''); url.searchParams.set('token', params.get('token') || ''); }
    ws = new WebSocket(url);
    ws.onopen = () => { $('mode').textContent = isPlayer ? 'PLAYER' : 'LIVE'; status('Waiting for both players.'); };
    ws.onmessage = event => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'hello') {
          slot = message.slot; names(message.players); $('play-controls').hidden = false;
        } else if (message.type === 'observation') observe(message);
        else if (message.type === 'global') {
          names(message.players); scoreboard(message.turn, message.army, message.land);
          if (message.phase === 'finished') { loadReplay(livePrefix+'/replay.json').catch(err => fail(err.message)); }
          else {
            $('cover-title').textContent = message.phase === 'waiting' ? 'Waiting for the generals' : message.phase === 'failed' ? 'Match could not finish' : 'The battle is underway';
            $('cover-text').textContent = 'The live map stays hidden to protect fog of war. Watch the score here, then explore the full replay.';
            status(message.phase === 'waiting' ? 'Waiting for both players.' : message.phase === 'failed' ? 'The episode failed. Check the game logs.' : 'Live score · Board revealed after the match.'); ready();
          }
        } else if (message.type === 'final') {
          sent = true; clearTimeout(passTimer); status(outcome(message.result));
          loadReplay(livePrefix+'/replay.json').catch(err => fail(err.message));
        } else if (message.type === 'error') status(message.message, true);
        else if (message.type === 'failure') fail('The episode could not complete.');
      } catch (err) { fail(err.message); }
    };
    ws.onerror = () => status('Connection failed. Check the player link and game server.', true);
    ws.onclose = () => {
      clearTimeout(passTimer); queue.length = 0; inFlight = null; sent = true; draw();
      if (!replay) status('Connection closed. Reopen your player link to reconnect.', true);
    };
  }
  new ResizeObserver(draw).observe(boardElement.parentElement);
  window.addEventListener('hashchange', () => {
    const url = new URLSearchParams(location.hash.slice(1)).get('replay');
    if (url) loadReplay(url).catch(err => fail(err.message));
  });
  if (replayURL) loadReplay(replayURL).catch(err => fail(err.message));
  else if (location.pathname.endsWith('/client/replay')) loadReplay(livePrefix+'/replay.json').catch(err => fail(err.message));
  else if (location.pathname.includes('/client/')) connectLive();
  else fail('Open this viewer with a replay URL in #replay=.');
})();
