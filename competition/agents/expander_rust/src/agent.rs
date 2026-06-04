// Edit this file to implement your agent.
//
// `Observation` is the parsed per-turn view from main.rs.
// `Agent::act` is called once per turn and must return an `Action`:
//
//     pass:      1 to skip the turn, 0 to move
//     row, col:  source cell
//     dir:       0=up, 1=down, 2=left, 3=right
//     split:     0=move all-but-one armies, 1=move half (floor)
//
// Tile-type codes in obs.type_grid:
//     0=fog, 1=plain, 2=mountain, 3=city, 4=general, 5=structure-in-fog
//
// Owner codes in obs.owner_grid (perspective-relative):
//     0=neutral/unknown, 1=me, 2=opp

pub struct Observation {
    pub h: usize,
    pub w: usize,
    pub turn: i32,
    pub my_land: i32,
    pub my_army: i32,
    pub opp_land: i32,
    pub opp_army: i32,
    pub type_grid: Vec<i32>,    // h*w row-major: type_grid[r * w + c]
    pub owner_grid: Vec<i32>,
    pub army_grid: Vec<i32>,
}

#[derive(Copy, Clone)]
pub struct Action {
    pub pass: i32,
    pub row: i32,
    pub col: i32,
    pub dir: i32,
    pub split: i32,
}

const PASS: Action = Action { pass: 1, row: 0, col: 0, dir: 0, split: 0 };

pub struct Agent {
    pub player_id: i32,
    pub h: usize,
    pub w: usize,
}

impl Agent {
    pub fn new(player_id: i32, h: usize, w: usize) -> Self {
        Self { player_id, h, w }
    }

    /// Expander: each turn pick the move that maximizes
    ///   score = src_army * (10 if expansion else 1) * (2 if opponent else 1)
    /// among captures (src_army > dest_army + 1). Otherwise the first valid
    /// move; otherwise pass.
    pub fn act(&self, obs: &Observation) -> Action {
        const DR: [i32; 4] = [-1, 1, 0, 0];   // up, down, left, right
        const DC: [i32; 4] = [ 0, 0, -1, 1];

        let h = obs.h as i32;
        let w = obs.w as i32;

        let mut best = PASS;
        let mut best_score: f64 = -1.0;
        let mut first_valid: Option<Action> = None;

        for r in 0..h {
            for c in 0..w {
                let idx = (r * w + c) as usize;
                if obs.owner_grid[idx] != 1 { continue; }
                let src_army = obs.army_grid[idx];
                if src_army <= 1 { continue; }

                for d in 0..4 {
                    let nr = r + DR[d];
                    let nc = c + DC[d];
                    if nr < 0 || nr >= h || nc < 0 || nc >= w { continue; }
                    let didx = (nr * w + nc) as usize;
                    let dtype = obs.type_grid[didx];
                    if dtype == 2 || dtype == 5 { continue; }   // impassable

                    let m = Action { pass: 0, row: r, col: c, dir: d as i32, split: 0 };
                    if first_valid.is_none() {
                        first_valid = Some(m);
                    }

                    let dest_owner = obs.owner_grid[didx];
                    let dest_army = obs.army_grid[didx];
                    if src_army <= dest_army + 1 { continue; }

                    let is_opp = dest_owner == 2;
                    let is_visible_neutral = dest_owner == 0 && dtype != 0 && dtype != 5;
                    let is_expansion = is_opp || is_visible_neutral;

                    let mut score = src_army as f64;
                    if is_expansion { score *= 10.0; }
                    if is_opp       { score *= 2.0; }

                    if score > best_score {
                        best_score = score;
                        best = m;
                    }
                }
            }
        }

        if best_score > 0.0 { best }
        else if let Some(fv) = first_valid { fv }
        else { PASS }
    }
}
