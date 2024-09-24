## TODOs

### Features
- [x] Let user pass argument indicating what is the default opponent agent for Gymnasium environment.
- [x] Extend action space by allowing user to be IDLE.
- [ ] Add human control, so people can play against bots.
- [x] Random agent new parameters

### Improvements
- [x] Let user change game speed even when the game is paused; make 'Paused' text separate in the side panel.
- [x] Revisit types for observation space (np.float32 vs np.bool)
- [x] Make ExpanderAgent a bit more readable if possible
- [x] Test IDLE actions

### Bug fixes

### Documentation and CI
- [ ] Create more examples of usage (Stable Baselines3 demo)
- [x] Pre-commit hooks for conventional commit checks (enforcing conventional commits)
- [x] Add CI for running tests (pre commit)
- [x] Add CI passing badge to README
