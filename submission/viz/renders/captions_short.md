# Short captions (two to three sentences per figure / table; computed by viz.short_captions)

## fig_games_per_year

Shows how many analysis games each calendar year contributes, stacked by time control. 78% of the 863,971 games are from 2024 or later, and classical makes up 71% of all games. Caveat: 2026 is a partial year (data cover Jan-Sep 2026), and the growth reflects how many events were broadcast on Lichess, not how much chess was played.

## fig_gap_distribution

Shows how far apart in rating the two players are, as the share of each time control's games per 25-point gap bin. The median gap is 152 points in classical, 188 in rapid and 174 in blitz, and only 8.3% of the 862,599 games with an underdog have a gap of 400 or more. Caveat: the gaps are between published FIDE ratings, which lag a player's current strength, especially for fast-improving juniors.

## fig_outcomes_by_gap

Shows how often the lower-rated player wins, draws and loses as the rating gap grows, per time control. In classical the underdog wins 25.9% of games in the 0-49 gap bin but only 5.8% at 400-449 (blitz: 37.0% vs 11.4%), and draws take up much of the slack in classical (42.2% draws at 0-49). Caveat: the curves pool both colours and all rating levels, which hides large subgroup differences (see the Simpson figure).

## fig_simpson

Asks whether one pooled classical curve hides subgroups, by splitting the underdog's results by its colour and by the rating band. At gaps of 50-199 the underdog scores 0.407 points per game with White but 0.328 with Black, and between near-equal players the draw rate rises from 33% (<2200) to 63% (2500+). Caveat: the band is defined by the average rating, so bands also differ in their gap mix and in who plays, and colour and band are shown one at a time.

## fig_heatmap

Shows the underdog's observed win rate in each cell of rating gap x average rating, per time control, to test whether the chance depends on the rating level as well as on the gap. In classical the win rate first falls below 10% at a median gap of 350 points for average ratings 1600-1999, but already at 200 for 2200-2599. Caveat: the green step is the first crossing of a noisy observed rate, not a fitted boundary, and draws count as not winning (cells with fewer than 100 games are hatched; the coloured cells hold 99.4% of the games).

## fig_set_compare

Compares the larger cleaned set (1,129,532 games) with the analysis set (863,971) to show whom the filters removed. Most of the 265,561 dropped games lack a FIDE rating for a player (238,168), and only 17% of the dropped games have a FIDE-titled player, so the share of games with a titled player rises from 45% to 54%. Caveat: the analysis set therefore over-represents titled, rated players, so the results describe rated tournament chess, not every broadcast game.

## fig_upset_points

Shows where the upsets at the 10% level (the underdog won, or drew, although the model gave it less than a 10% chance of doing so) fall in rating gap and average rating, with the 20 rarest numbered and listed. There are 11,114 upsets (9,007 classical, 1,483 rapid, 624 blitz; 604 are draws), and the 20 rarest are all classical wins with gaps of 626-1218 points. Caveat: the plot shows where upsets occur, not how likely they are (dense regions are dense mainly because many games are played there), and a very large gap often means an outdated rating rather than a sensational game.

## fig_model_vs_data

Checks bin by bin whether the model's predicted underdog win rate matches the observed rate, by rating band and time control. The prediction lies inside the observed 95% interval in 193 of 229 bins with at least 100 games (84%; about 95% would be expected for an exact model). Caveat: the clearest misfit is classical 2500+, where underdogs win less often than predicted (only 5 of 14 bins inside, mean error -2.0 percentage points), and the interaction model does not remove it (5 of 14 bins).

## fig_verified_split

Explains why the old notebook plot showed observed win rates far above the model curve (White underdog, 2200-2499): its dots used only the games whose ratings could not be verified against the FIDE list. In rapid at gap 250-299 those games show 21.6% (n=529), whereas all games show 17.8% and the verified games 15.9%, against 15.9% for the curve. Caveat: the grey 95% band reflects only the model's parameter uncertainty, so bins of a few hundred games often lie outside it even when the model is right.

## fig_critical_gap_alpha

Shows the critical gap (the smallest gap at which the model gives the underdog P(win) below alpha) for every alpha from 0.01 to 0.30, to test whether the ordering classical < rapid < blitz depends on the choice of alpha. The ordering holds at 295 of 306 grid points from 0.05 to 0.30 (the 11 exceptions are Black-underdog ties at gap 0 for alpha of 0.265 or more); at alpha = 0.10 and average rating 2200-2499 the critical gaps are 280 / 405 / 465 points (classical / rapid / blitz) for a White and 185 / 310 / 405 for a Black underdog. Caveat: at alpha = 0.05 the interaction model gives critical gaps 54 points smaller on average (only 4 of 18 inside the main interval), so the main model's alpha = 0.05 values should only be quoted with that caveat.

## table_critical_gaps

Lists the critical gap at alpha = 0.10 for every time control, underdog colour and rating band, with its 95% interval and whether it lies inside the observed data. In the 2200-2499 band an underdog's win counts as an upset from a gap of 280 / 405 / 465 points (classical / rapid / blitz) with White and 185 / 310 / 405 with Black. Caveat: 3 of the 18 values (rapid, white, 2500+; blitz, white, 2500+; blitz, black, 2500+) lie beyond the gaps observed in their slice, so they are extrapolations.

## table_differences

Gives the difference of the critical gap between time controls at alpha = 0.10 (bootstrap median with 95% interval), per underdog colour and rating band. All 18 intervals exclude 0, with differences from 60 to 250 points in size; for a White underdog at 2200-2499, classical minus blitz is -190 (-220 to -160). Caveat: the published difference is the median of bootstrap differences, so it can differ from the difference of the two point estimates (here by up to 5 points).

## table_robustness

Compares the critical gaps at alpha = 0.10 (band 2200-2499) across the main run, the verified-ratings run, the sensitivity run, a model-free isotonic cross-check and the interaction model. Across the four model runs each critical gap moves by at most 45 points (classical: 10), and the ordering classical < rapid < blitz is the same in every run. Caveat: the isotonic values are pooled over all rating bands, so they are not strictly comparable (with them the spread reaches 85 points).

## fig_upset_rule

Shows the critical gap at alpha = 0.10 as a continuous function of the average rating, per time control and underdog colour, i.e. the upset rule for any pair of players. For a White underdog in classical it falls from 375 points at an average rating of 1800 to 235 at 2600 (blitz: 510 to 450). Caveat: the decline comes from a single linear rating term, and in the model's tail 2500+ underdogs win only 0.77 times the expected number (0.78 with the interaction model), so the line is probably too high for strong players.

## fig_calibration_check

Tests on tournaments the model never saw (a held-out half of the tournaments, 426,692 games) whether predicted and observed underdog win rates agree. The mean absolute error is 0.39 percentage points (n-weighted, over 48 bins), and the largest bin error is -2.2 percentage points (blitz, black underdog, gap 600-699, n = 373). Caveat: the bins pool all rating levels, which hides the strong-player misfit (fig_model_misfit), and this is a single random split.

## table_expected_upsets

Compares the number of upsets a calibrated model would expect (the sum of its own probabilities) with the number observed. At alpha = 0.10 the model expects 11,273 upsets and 11,114 are observed (ratio 0.99); at alpha = 0.05 it is 1,344 vs 1,183 (0.88). Caveat: the z-values use a binomial standard deviation that ignores tournament clustering and parameter uncertainty, so they overstate significance.

## fig_model_misfit

Asks whether the model's tail is right for strong players, and whether adding rating x gap terms (the interaction model) fixes it and moves the critical gaps. Among games the model gives P(win) < 0.10, underdogs at 2500+ win only 0.77 times the expected number (interaction model 0.78), while the 2200-2499 tail is repaired (0.91 to 0.97). Caveat: at alpha = 0.05 the interaction model makes the critical gaps 54 points smaller on average (only 4 of 18 inside the main interval, vs 16 of 18 at 0.10), so the main model's alpha = 0.05 critical gaps are probably too large.

## table_model_misfit

Lists, by rating band, the underdog wins expected and observed among the games each model gives P(win) < alpha, plus the bin-by-bin fit by band. At alpha = 0.10 and 2500+ the main model expects 664 wins and 512 are observed (ratio 0.77); the interaction model gives 0.78. Caveat: the approximate 95% intervals ignore tournament clustering and parameter uncertainty, so they are too narrow.

## table_alpha005_caveat

Lists the critical gaps of the main and the interaction model at alpha 0.20 / 0.10 / 0.05 for every cell, with the change interaction - main. At alpha = 0.10 the changes range -40 to +15 (mean -6; 16 of 18 inside the main 95% interval), at alpha = 0.05 -120 to -10 (mean -54; 4 of 18 inside the main 95% interval). Caveat: at alpha = 0.05 most interaction values fall outside the main model's interval, so the main model's alpha = 0.05 critical gaps are probably too large and should only be quoted with this caveat (the alpha = 0.10 results are robust to the model form).

