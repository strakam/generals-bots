# Changelog

## [0.5.0](https://github.com/strakam/Generals-RL/compare/v0.4.0...v0.5.0) (2024-10-01)


### 🚀 Features

* Reset replay if 'r' is pressed ([0c2357f](https://github.com/strakam/Generals-RL/commit/0c2357f2f49493d7b17804dc1144ae1b1c1fba80))


### 🐛 Bug Fixes

* Accept map passed from options ([baf2021](https://github.com/strakam/Generals-RL/commit/baf202193686b9cd97e4dcca9a611b4e4af2fb08))
* Create proper types for spaces ([f543663](https://github.com/strakam/Generals-RL/commit/f543663c0a8f332043fd8ec4ea86c11ab84893cf))


### 🛠️ Refactor

* add missing types ([960afae](https://github.com/strakam/Generals-RL/commit/960afae4080382fceffa06a66e882108a38f716a))
* move common attributes to Agent class ([e48a39b](https://github.com/strakam/Generals-RL/commit/e48a39bd91fcd60c9053e925822d820e349ab94e))
* New action space and observation space ([4979342](https://github.com/strakam/Generals-RL/commit/49793423e9039bae463d21f80c0d4923a7f44d4b))
* Unify reward_fn design ([3039ace](https://github.com/strakam/Generals-RL/commit/3039aceb27d63cdbbfe8f8f8fee8d71451e6915e))

## [0.4.0](https://github.com/strakam/Generals-RL/compare/v0.3.2...v0.4.0) (2024-09-29)


### 🚀 Features

* Support rectangular grid shapes ([bb3c5e1](https://github.com/strakam/Generals-RL/commit/bb3c5e1ef7cd53c8d92c3a901010ef6f263e73d8))


### 🐛 Bug Fixes

* Delete replay class on reset ([3ee9a68](https://github.com/strakam/Generals-RL/commit/3ee9a688a73eacdd5a780baeb4186edeaf4ddf06))
* Fix possibility that two generals are generated in the same location ([d63978b](https://github.com/strakam/Generals-RL/commit/d63978b2b09acda1b22d86d80386117e508e7ff2))

## [0.3.2](https://github.com/strakam/Generals-RL/compare/v0.3.1...v0.3.2) (2024-09-27)


### 🐛 Bug Fixes

* **mapper:** properly regenerate invalid map ([63ff837](https://github.com/strakam/Generals-RL/commit/63ff837c86d57f6b92d7445b5f1aaff0aed43673))


### 🛠️ Refactor

* Change agent passing, minimize examples ([11a38b7](https://github.com/strakam/Generals-RL/commit/11a38b711816e33300f25c4f024653f813b1be49))
* **render:** simplify code ([58e8b01](https://github.com/strakam/Generals-RL/commit/58e8b0142c2e4e775488db00e2e8205d618b9952))

## [0.3.1](https://github.com/strakam/Generals-RL/compare/v0.3.0...v0.3.1) (2024-09-26)


### 🛠️ Refactor

* gigantic refactor ([d67e887](https://github.com/strakam/Generals-RL/commit/d67e8877523850df0c017fea0b5d7b7dd1dd92ae))
* Rethink maps ([bbfb777](https://github.com/strakam/Generals-RL/commit/bbfb777cd4d02ba2f415171fdd94e74680fa80d8))
* rethink replay ([ebc2587](https://github.com/strakam/Generals-RL/commit/ebc2587ade5e2d3ecab11177a0c428406fc053af))

## [0.3.0](https://github.com/strakam/Generals-RL/compare/v0.2.1...v0.3.0) (2024-09-25)


### 🚀 Features

* add custom colors and names ([8fcc858](https://github.com/strakam/Generals-RL/commit/8fcc8584702fc975548cbb2b6894db41272a8fee))


### 🐛 Bug Fixes

* **game:** check validity of moves in game step ([e6fc5f0](https://github.com/strakam/Generals-RL/commit/e6fc5f0a2cb07ddf630812ae965c3b932edbbdde))
* only non-trivial actions are valid ([e3119cd](https://github.com/strakam/Generals-RL/commit/e3119cd5f7182d50619859f5ebdf867bc1257f82))
* raise warning when invalid moves are made by agents ([6cd5257](https://github.com/strakam/Generals-RL/commit/6cd52572c2aeff50d567034021a33983acd00153))


### 🛠️ Refactor

* **agent:** cleaner code, better naming ([a4b3b64](https://github.com/strakam/Generals-RL/commit/a4b3b6426582d668068cbfa13978939e76674679))
* **game:** change move format ([f14d716](https://github.com/strakam/Generals-RL/commit/f14d7165e6f859604858d2dd046b10b3c702b702))
* Simplify Expander code ([4cedc9f](https://github.com/strakam/Generals-RL/commit/4cedc9fddbc4ecb67516399046786d849645e00d))

## [0.2.1](https://github.com/strakam/Generals-RL/compare/v0.1.0...v0.2.1) (2024-09-24)


### Features

* add parameters to random agent ([d3451b7](https://github.com/strakam/Generals-RL/commit/d3451b7b64f9377301b65286cc40d7f3f758591c))
* extend action space (idle action allowed) ([9d4e1fa](https://github.com/strakam/Generals-RL/commit/9d4e1fae508f5c419a77d77c17dd7409e0e5bfc2))
* user can choose gymnasium opponent NPC ([6ff2120](https://github.com/strakam/Generals-RL/commit/6ff212064a2aa613812bcbec021d6947c9b04cc9))


### Bug Fixes

* allow change of game speed during pause ([99606c8](https://github.com/strakam/Generals-RL/commit/99606c8b477368e0d617556ccd226ddd6f7e168a))
* make info dictionaries empty ([7c1e0c0](https://github.com/strakam/Generals-RL/commit/7c1e0c0688a5c181b670bfdc52f5ce34d7b54924))
