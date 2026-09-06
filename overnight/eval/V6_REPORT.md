# v6 candidate -- 2026-09-05 03:24

Switches on: LMR ASPIRATION SEE REPETITION_TWOFOLD

- 052-lmr: PROMOTE  052-lmr  +47 Elo over 232 games
- 051-pvs: REJECT  051-pvs  no better than the champion
- 053-lmp: REJECT  053-lmp  no better than the champion
- 054-aspiration: PROMOTE  054-aspiration  +41 Elo over 237 games
- 055-see: PROMOTE  055-see  +25 Elo over 478 games
- 056-kz32: PROMOTE  056-kz32  +70 Elo over 136 games
- 059-kz32b: INCONCLUSIVE  059-kz32b  ran out of games; champion stays
- 052b-lmr: PROMOTE  052b-lmr  +46 Elo over 196 games
- 057-twofold: PROMOTE  057-twofold  +66 Elo over 160 games
- 060-v6: PROMOTE  060-v6  +90 Elo over 167 games

Crash hunt:   terminations: adjudication 6, checkmate 335, fifty_moves 12, insufficient_material 60, stalemate 4, threefold_repetition 83
Clock replay: flags 0/6  errors 0  lowest clock 10.2s  longest move 10.1s -> PASS
120 s vs 050:   +14 =17 -9  score 56.2%  over 40 games

Training (kz32b):   restored the best epoch, validation loss 0.004690 wrote training\checkpoints\net_w512-b8-kz32b.json: {'best_val': 0.004690001923424673, 'initial_val': 0.004842887690514963, 'epochs': 24.0} 
NNUE check: all checks passed
