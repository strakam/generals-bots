// C++ starter — game-loop scaffolding.
//
// You shouldn't need to edit this file. Implement your strategy in agent.hpp.
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
//     Game end: the engine closes our stdin. We exit on EOF (scanf returns
//     EOF, which is != the expected token count).
//
// scanf skips whitespace (spaces, tabs, newlines) between integers, so we
// don't need to handle the line structure explicitly -- just read ints.
//
// fflush(stdout) after every write is mandatory: pipes are fully buffered
// by default, so without it the runner never sees our action and both
// sides deadlock.
#include <cstdio>
#include <vector>

#include "agent.hpp"


static bool read_grid(std::vector<int>& g, int N) {
    for (int i = 0; i < N; ++i) {
        if (scanf("%d", &g[i]) != 1) return false;
    }
    return true;
}

int main() {
    int player_id, H, W;
    if (scanf("%d %d %d", &player_id, &H, &W) != 3) return 0;
    const int N = H * W;

    Agent agent(player_id, H, W);

    Observation obs;
    obs.H = H;
    obs.W = W;
    obs.type_grid.resize(N);
    obs.owner_grid.resize(N);
    obs.army_grid.resize(N);

    while (true) {
        if (scanf("%d", &obs.turn) != 1) return 0;   // EOF -> game over

        if (scanf("%d %d %d %d",
                  &obs.my_land, &obs.my_army,
                  &obs.opp_land, &obs.opp_army) != 4) return 0;

        if (!read_grid(obs.type_grid,  N)) return 0;
        if (!read_grid(obs.owner_grid, N)) return 0;
        if (!read_grid(obs.army_grid,  N)) return 0;

        Action a = agent.act(obs);
        printf("%d %d %d %d %d\n", a.pass, a.row, a.col, a.dir, a.split);
        fflush(stdout);
    }
}
