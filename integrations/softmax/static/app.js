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
  let ruleset = 'classic', playerCount = 2, eliminated = false;
  const moves = [[-1,0],[1,0],[0,-1],[0,1]], queue = [];
  let inFlight = null, selectedRoute = null, nextRoute = 0;
  function selectTile(tile, route = null) {
    selected = tile; selectedRoute = tile ? route ?? ++nextRoute : null;
  }
  const sameTile = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];
  const canQueue = () => slot !== null && board && ws?.readyState === WebSocket.OPEN && !replay && !eliminated;
  // Fog obstacles may hide castles or mountains. Plan around them until revealed.
  const blocked = (r, c) => r < 0 || c < 0 || r >= board.type_grid.length ||
    c >= board.type_grid[0].length || [2, 5].includes(board.type_grid[r][c]);
  function blockedMove(action) {
    if (action[0] !== 0) return false;
    const [, r, c, direction] = action, [dr, dc] = moves[direction];
    return blocked(r+dr, c+dc);
  }
  const post = data => { if (parent !== window) parent.postMessage({src: 'coworld-replay', ...data}, '*'); };
  function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
  function ready() { if (!readySent) { readySent = true; setTimeout(() => post({type: 'ready'}), 0); } }
  function fail(message) {
    status(message, true); $('cover').hidden = false;
    $('cover-title').textContent = 'Unable to open this match'; $('cover-text').textContent = message;
    post({type: 'error', message});
  }
  function names(players) {
    const matchbar = document.querySelector('.matchbar');
    playerCount = players.length;
    matchbar.classList.toggle('ffa', playerCount === 4);
    for (let i = 2; i < 4; i++) {
      if (i >= playerCount) { $(`name${i}`)?.closest('.player').remove(); continue; }
      if (!$(`name${i}`)) {
        const card = document.createElement('div'); card.className = `player ${i === 2 ? 'green' : 'purple'}`;
        card.innerHTML = `<div><b id="name${i}"></b><small><span id="army${i}">—</span> army <span class="divider">/</span> <span id="land${i}">—</span> land</small></div>`;
        matchbar.append(card);
      }
    }
    players.forEach((name, i) => { if ($(`name${i}`).textContent !== name) $(`name${i}`).textContent = name; });
    const title = ruleset === 'build_castles' ? 'BUILD YOUR CASTLES' : playerCount === 4 ? '4-PLAYER FFA' : 'CLASSIC 1v1';
    $('variant-title').textContent = title; document.title = `Generals · ${title}`;
  }
  function scoreboard(turn, army, land, eliminatedPlayers = []) {
    $('turn').textContent = turn;
    for (let i = 0; i < playerCount; i++) {
      $(`army${i}`).textContent = army[i] ?? '—'; $(`land${i}`).textContent = land[i] ?? '—';
      $(`name${i}`).closest('.player').classList.toggle('eliminated', !!eliminatedPlayers[i]);
    }
    $('endgame').textContent = ruleset === 'competition'
      ? (turn >= 800 ? 'DEATHTOUCH ACTIVE · Reach the enemy general to win' : 'Deathtouch from turn 800')
      : ruleset === 'build_castles' ? 'Build castles on your land · Cost starts at 35 army'
      : playerCount === 4 ? 'Capture generals · Last general standing wins' : 'Capture castles to grow your army';
  }
  function buildProblem(r, c) {
    const cost = board.build_cost_grid?.[r]?.[c] || 0;
    if (board.owner_grid[r][c] !== slot+1 || board.type_grid[r][c] !== 1 || cost <= 0) return 'select your own plain land';
    if (board.army_grid[r][c] < cost) return `need ${cost} army to build here (${board.army_grid[r][c]} available)`;
    return null;
  }
  function updateBuildControl() {
    const enabled = ruleset === 'build_castles' && slot !== null && !replay;
    if (enabled && !$('build')) {
      const button = document.createElement('button'); button.id = 'build';
      button.innerHTML = 'Build castle <kbd>B</kbd>'; button.onclick = buildCastle;
      $('play-controls').insertBefore(button, $('pass'));
    } else if (!enabled) $('build')?.remove();
    $('build-hint').hidden = !enabled;
    if (enabled) {
      const cost = selected && board?.build_cost_grid?.[selected[0]]?.[selected[1]];
      $('build').disabled = !canQueue() || !selected || !!buildProblem(...selected);
      $('build-hint').textContent = eliminated ? 'You have been eliminated.' : cost
        ? `Castle cost: ${cost} army · ${board.army_grid[selected[0]][selected[1]]} on this tile`
        : 'Select your own plain land to see its castle cost.';
    }
  }
  function buildCastle() {
    if (!canQueue() || ruleset !== 'build_castles') return;
    if (!selected) { status('Select your own plain land to build a castle.'); return; }
    const problem = buildProblem(...selected);
    if (problem) { status(`Cannot build: ${problem}.`); return; }
    enqueue([2, ...selected, 0, 0]);
  }
  function draw() {
    if (!board) return;
    renderer.draw(board, selected, queue.map(item => item.action), inFlight?.action);
    $('queue-count').textContent = `${queue.length} queued`;
    updateBuildControl();
    $('cover').hidden = true; ready();
  }
  function send(action, automatic = false) {
    if (!ws || ws.readyState !== WebSocket.OPEN || sent || currentTurn < 0) return false;
    ws.send(JSON.stringify({type: 'action', turn: currentTurn, action}));
    sent = true; clearTimeout(passTimer);
    if (!automatic) status(action[0] === 1 ? 'Holding position.' : action[0] === 2 ? 'Castle build submitted.' : 'Move submitted.');
    return true;
  }
  function dispatchQueue() {
    if (!canQueue() || sent) return;
    while (queue.length) {
      const item = queue[0], [kind, r, c] = item.action;
      let reason;
      if (kind !== 1 && board.owner_grid[r][c] !== slot+1) reason = 'the next source tile is no longer yours';
      else if (kind === 0 && blockedMove(item.action)) reason = 'the next move is blocked';
      else if (kind === 0 && board.army_grid[r][c] < 2) reason = 'not enough army to move';
      else if (kind === 2) reason = buildProblem(r, c);
      if (reason) {
        stopRoute(item.route, reason);
        continue;
      }
      if (send(item.action)) inFlight = queue.shift();
      break;
    }
    draw();
  }
  function stopRoute(route, reason) {
    for (let i = queue.length-1; i >= 0; i--) if (queue[i].route === route) queue.splice(i, 1);
    if (inFlight?.route === route) inFlight = null;
    if (selectedRoute === route) selectTile(null);
    draw(); status(`Route stopped: ${reason}.${queue.length ? ' Continuing with the next queued route.' : ' Select a tile to start a new route.'}`);
  }
  function enqueue(action, to = selected) {
    if (!canQueue()) return;
    queue.push({action, from: selected, to, route: selectedRoute ?? ++nextRoute}); selected = to;
    status(`${queue.length} queued · E undoes one, Q clears the queue.`);
    dispatchQueue(); draw();
  }
  function undoMove() {
    const removed = queue.pop();
    if (removed && selectedRoute === removed.route && sameTile(selected, removed.to)) selectTile(removed.from, removed.route);
    draw(); status(removed ? 'Last queued action removed.' : 'No queued moves to undo.');
  }
  function clearQueue() {
    if (queue.length && selectedRoute === queue[queue.length-1].route && sameTile(selected, queue[queue.length-1].to)) {
      selectTile(queue[0].from, queue[0].route);
    }
    queue.length = 0; draw();
    status(inFlight ? 'Queue cleared. The submitted action will finish this turn.' : 'Queue cleared.');
  }
  function move(direction) {
    if (!selected || !canQueue()) return;
    const [r, c] = selected, delta = moves[direction];
    const nr = r+delta[0], nc = c+delta[1];
    if (blocked(nr, nc)) {
      status('Route blocked · Choose a direction around the obstacle.'); return;
    }
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
    if (board.owner_grid[r][c] === slot+1) selectTile([r,c]);
    draw(); boardElement.focus();
  });
  $('split').onclick = () => { half = !half; $('split').setAttribute('aria-pressed', String(half)); };
  $('pass').onclick = () => enqueue([1,0,0,0,0]);
  $('deselect').onclick = () => { selectTile(null); draw(); status('Selection cleared.'); };
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
    else if (key === 'b' && ruleset === 'build_castles') { event.preventDefault(); buildCastle(); }
    else if (key === 'e') { event.preventDefault(); undoMove(); }
    else if (key === 'q') { event.preventDefault(); clearQueue(); }
    else if (key === ' ') { event.preventDefault(); $('deselect').click(); }
  });
  function outcome(result) {
    const text = result.winner < 0 ? 'Draw' : `${$('name'+result.winner).textContent} wins`;
    const reasons = {general_capture:'general captured', turn_limit:'turn limit reached', forfeit:'timeout elimination', double_forfeit:'remaining players timed out'};
    return `${text} · ${reasons[result.reason] || result.reason}`;
  }
  function showReplayFrame() {
    board = replay.frames[frameIndex]; scoreboard(board.turn, board.army, board.land, board.eliminated); draw();
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
    eliminated = false; updateBuildControl();
    queue.length = 0; inFlight = null; selectTile(null);
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
        !Array.isArray(data.players) || ![2,4].includes(data.players.length) || !data.result) throw new Error('Unsupported or incomplete Generals replay.');
    for (const frame of data.frames) for (const key of ['type_grid','owner_grid','army_grid']) {
      if (!Array.isArray(frame[key]) || frame[key].length !== data.height || frame[key].some(row => !Array.isArray(row) || row.length !== data.width || row.some(n => !Number.isInteger(n)))) throw new Error('Replay contains an invalid board.');
    }
    replay = data; ruleset = data.ruleset || 'competition'; slot = null; selected = null; clearTimeout(passTimer);
    names(data.players); $('mode').textContent = 'REPLAY';
    $('play-controls').hidden = true; $('replay-controls').hidden = false; $('seek').max = data.frames.length-1;
    post({type:'phase', phase:'replay_parsed'});
    readySent = false; frameIndex = 0; showReplayFrame(); resetReplayTimer();
  }
  function observe(message) {
    const firstTurn = currentTurn < 0;
    const previousMove = inFlight;
    slot = message.slot; currentTurn = message.turn; sent = false; inFlight = null;
    ruleset = message.ruleset || ruleset; eliminated = !!message.eliminated; names(message.players);
    if (playerCount > 2 && !message.visible_owner_grid) throw new Error('This multiplayer observation is missing visible owner identities.');
    const owners = message.visible_owner_grid || message.owner_grid.map(row => row.map(o => o === 0 ? 0 : o === 1 ? slot+1 : 2-slot));
    board = {...message, owner_grid: owners};
    // A failed step invalidates its route, but independently selected armies
    // keep their plans and can take over this turn.
    const failedMove = previousMove?.action[0] === 0 &&
      (message.last_move_executed === false || owners[previousMove.to[0]][previousMove.to[1]] !== slot+1);
    const failedBuild = previousMove?.action[0] === 2 && message.last_build_executed === false;
    const obstructedRoutes = new Set(queue.filter(item => blockedMove(item.action)).map(item => item.route));
    if (!queue.length && selected && owners[selected[0]][selected[1]] !== slot+1) selectTile(null);
    const army = message.public_scores?.army || [], land = message.public_scores?.land || [];
    if (!message.public_scores && playerCount === 2) {
      army[slot] = message.my_army; army[1-slot] = message.opp_army;
      land[slot] = message.my_land; land[1-slot] = message.opp_land;
    }
    scoreboard(message.turn, army, land, message.public_scores?.eliminated); draw();
    if (firstTurn) status('Select a tile and queue moves with the arrow keys.');
    clearTimeout(passTimer);
    if (eliminated) {
      queue.length = 0; selectTile(null); sent = true; draw();
      $('play-controls').hidden = true;
      status('You have been eliminated. Follow the scores until the match ends, then watch the full replay.');
      return;
    }
    // The deadline starts on the server, before either leg of the proxy trip.
    // Queue inputs made after this pass will execute on the following tick.
    passTimer = setTimeout(() => send([1,0,0,0,0], true), Math.min(50, message.turn_timeout_seconds * 200));
    if (failedMove) stopRoute(previousMove.route, 'the previous move did not reach its destination');
    if (failedBuild) stopRoute(previousMove.route, 'the castle could not be built');
    else if (previousMove?.action[0] === 2 && message.last_build_executed === true) status('Castle built.');
    for (const route of obstructedRoutes) stopRoute(route, 'an obstacle blocks the route');
    dispatchQueue();
  }
  function connectLive() {
    const isPlayer = location.pathname.endsWith('/client/player');
    const route = livePrefix + (isPlayer ? '/player' : '/global');
    const address = params.get('address');
    let url;
    if (address !== null) {
      try {
        url = new URL(address);
        if (!['ws:', 'wss:'].includes(url.protocol) || url.host !== location.host || url.pathname !== route) {
          throw new Error('Invalid connection address');
        }
      } catch (_) { fail('Invalid connection address. Reopen this game from its lobby.'); return; }
    } else {
      url = new URL(route, location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      if (isPlayer) { url.searchParams.set('slot', params.get('slot') || ''); url.searchParams.set('token', params.get('token') || ''); }
    }
    ws = new WebSocket(url);
    ws.onopen = () => { $('mode').textContent = isPlayer ? 'PLAYER' : 'LIVE'; status('Waiting for all players.'); };
    ws.onmessage = event => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'hello') {
          ruleset = message.ruleset || 'classic';
          slot = message.slot; names(message.players); $('play-controls').hidden = false;
          updateBuildControl();
        } else if (message.type === 'observation') observe(message);
        else if (message.type === 'global') {
          ruleset = message.ruleset || ruleset;
          names(message.players); scoreboard(message.turn, message.army, message.land, message.eliminated);
          if (message.phase === 'finished') { loadReplay(livePrefix+'/client/replay.json').catch(err => fail(err.message)); }
          else {
            $('cover-title').textContent = message.phase === 'waiting' ? 'Waiting for the generals' : message.phase === 'failed' ? 'Match could not finish' : 'The battle is underway';
            $('cover-text').textContent = 'The live map stays hidden to protect fog of war. Watch the score here, then explore the full replay.';
            status(message.phase === 'waiting' ? 'Waiting for all players.' : message.phase === 'failed' ? 'The episode failed. Check the game logs.' : 'Live score · Board revealed after the match.'); ready();
          }
        } else if (message.type === 'final') {
          sent = true; clearTimeout(passTimer); status(outcome(message.result));
          loadReplay(livePrefix+'/client/replay.json').catch(err => fail(err.message));
        } else if (message.type === 'error') {
          if (inFlight) stopRoute(inFlight.route, message.message);
          else status(message.message, true);
        }
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
  else if (location.pathname.endsWith('/client/replay')) loadReplay(livePrefix+'/client/replay.json').catch(err => fail(err.message));
  else if (location.pathname.includes('/client/')) connectLive();
  else fail('Open this viewer with a replay URL in #replay=.');
})();
