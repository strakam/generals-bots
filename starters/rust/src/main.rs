// Rust starter — game-loop scaffolding.
//
// You shouldn't need to edit this file. Implement your strategy in agent.rs.
//
// Protocol summary (see generals/protocol.py for the full spec):
//     Handshake (engine -> agent, once):
//         <player_id> <H> <W>
//     Per turn (engine -> agent):
//         <turn> <my_land> <my_army> <opp_land> <opp_army>
//         H lines of W ints   (type grid)
//         H lines of W ints   (owner grid)
//         H lines of W ints   (army grid)
//     Per turn (agent -> engine):
//         <pass> <row> <col> <dir> <split>
//     Game end: the engine closes our stdin. We exit on EOF (next_int
//     returns None).
//
// The Tokenizer treats stdin as a whitespace-delimited int stream so we
// don't have to care about line boundaries. stdout.flush() after every
// write is mandatory: pipes are fully buffered by default, so without it
// the runner never sees our action and both sides deadlock.

mod agent;

use agent::{Agent, Observation};
use std::io::{self, BufRead, Write};


struct Tokenizer<R: BufRead> {
    reader: R,
    buf: String,
    pos: usize,
}

impl<R: BufRead> Tokenizer<R> {
    fn new(reader: R) -> Self {
        Self { reader, buf: String::new(), pos: 0 }
    }

    fn next_int(&mut self) -> Option<i32> {
        loop {
            let bytes = self.buf.as_bytes();
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos < bytes.len() {
                let start = self.pos;
                while self.pos < bytes.len() && !bytes[self.pos].is_ascii_whitespace() {
                    self.pos += 1;
                }
                return self.buf[start..self.pos].parse().ok();
            }
            self.buf.clear();
            self.pos = 0;
            match self.reader.read_line(&mut self.buf) {
                Ok(0) => return None,        // EOF
                Ok(_) => continue,
                Err(_) => return None,
            }
        }
    }
}


fn read_n(tok: &mut Tokenizer<impl BufRead>, n: usize) -> Option<Vec<i32>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(tok.next_int()?);
    }
    Some(v)
}


fn main() {
    let stdin = io::stdin();
    let mut tok = Tokenizer::new(stdin.lock());
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Handshake.
    let player_id = match tok.next_int() { Some(v) => v, None => return };
    let h = match tok.next_int() { Some(v) => v as usize, None => return };
    let w = match tok.next_int() { Some(v) => v as usize, None => return };
    let n = h * w;

    let agent = Agent::new(player_id, h, w);

    loop {
        let turn = match tok.next_int() { Some(v) => v, None => return };

        let my_land = match tok.next_int()  { Some(v) => v, None => return };
        let my_army = match tok.next_int()  { Some(v) => v, None => return };
        let opp_land = match tok.next_int() { Some(v) => v, None => return };
        let opp_army = match tok.next_int() { Some(v) => v, None => return };

        let type_grid = match read_n(&mut tok, n)  { Some(v) => v, None => return };
        let owner_grid = match read_n(&mut tok, n) { Some(v) => v, None => return };
        let army_grid = match read_n(&mut tok, n)  { Some(v) => v, None => return };

        let obs = Observation {
            h, w, turn,
            my_land, my_army, opp_land, opp_army,
            type_grid, owner_grid, army_grid,
        };

        let a = agent.act(&obs);
        writeln!(out, "{} {} {} {} {}", a.pass, a.row, a.col, a.dir, a.split).ok();
        out.flush().ok();
    }
}
