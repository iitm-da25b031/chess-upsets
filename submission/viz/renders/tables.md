## table_critical_gaps

| underdog | band (ref Elo) | time control | critical gap (95% CI) | data reach gap | data support |
|---|---|---|---|---|---|
| white | <2200 (2100) | classical | 320 (315-330) | 900 | inside |
| white | <2200 (2100) | rapid | 425 (405-450) | 900 | inside |
| white | <2200 (2100) | blitz | 485 (460-515) | 800 | inside |
| white | 2200-2499 (2350) | classical | 280 (270-285) | 650 | inside |
| white | 2200-2499 (2350) | rapid | 405 (385-425) | 650 | inside |
| white | 2200-2499 (2350) | blitz | 465 (440-495) | 650 | inside |
| white | 2500+ (2600) | classical | 235 (225-240) | 350 | inside |
| white | 2500+ (2600) | rapid | 385 (365-405) | 300 | BEYOND the data |
| white | 2500+ (2600) | blitz | 450 (425-475) | 350 | BEYOND the data |
| black | <2200 (2100) | classical | 235 (235-240) | 900 | inside |
| black | <2200 (2100) | rapid | 345 (335-360) | 900 | inside |
| black | <2200 (2100) | blitz | 430 (405-455) | 800 | inside |
| black | 2200-2499 (2350) | classical | 185 (180-185) | 700 | inside |
| black | 2200-2499 (2350) | rapid | 310 (300-325) | 650 | inside |
| black | 2200-2499 (2350) | blitz | 405 (380-430) | 650 | inside |
| black | 2500+ (2600) | classical | 125 (120-130) | 350 | inside |
| black | 2500+ (2600) | rapid | 280 (265-295) | 300 | inside |
| black | 2500+ (2600) | blitz | 380 (355-405) | 350 | BEYOND the data |

## table_differences

| underdog | band | pair | difference (median of draws, 95% interval) | point(a) - point(b) | CI excludes 0 |
|---|---|---|---|---|---|
| white | <2200 | classical - rapid | -105 (-125 to -85) | -105 | yes |
| white | <2200 | rapid - blitz | -60 (-95 to -25) | -60 | yes |
| white | <2200 | classical - blitz | -165 (-195 to -135) | -165 | yes |
| white | 2200-2499 | classical - rapid | -125 (-150 to -105) | -125 | yes |
| white | 2200-2499 | rapid - blitz | -60 (-95 to -30) | -60 | yes |
| white | 2200-2499 | classical - blitz | -190 (-220 to -160) | -185 | yes |
| white | 2500+ | classical - rapid | -150 (-170 to -130) | -150 | yes |
| white | 2500+ | rapid - blitz | -65 (-100 to -35) | -65 | yes |
| white | 2500+ | classical - blitz | -215 (-245 to -190) | -215 | yes |
| black | <2200 | classical - rapid | -110 (-125 to -95) | -110 | yes |
| black | <2200 | rapid - blitz | -85 (-115 to -55) | -85 | yes |
| black | <2200 | classical - blitz | -190 (-220 to -170) | -195 | yes |
| black | 2200-2499 | classical - rapid | -130 (-145 to -115) | -125 | yes |
| black | 2200-2499 | rapid - blitz | -90 (-120 to -65) | -95 | yes |
| black | 2200-2499 | classical - blitz | -220 (-245 to -200) | -220 | yes |
| black | 2500+ | classical - rapid | -155 (-170 to -140) | -155 | yes |
| black | 2500+ | rapid - blitz | -100 (-125 to -70) | -100 | yes |
| black | 2500+ | classical - blitz | -250 (-275 to -230) | -255 | yes |

## table_robustness

| time control | underdog | main | verified ratings | sensitivity run | isotonic (all bands) | interaction model | spread (all 5) | spread (4 model runs) |
|---|---|---|---|---|---|---|---|---|
| classical | white | 280 | 270 | 280 | 345 | 270 | 75 | 10 |
| rapid | white | 405 | 370 | 400 | 420 | 395 | 50 | 35 |
| blitz | white | 465 | 420 | 450 | 450 | 440 | 45 | 45 |
| classical | black | 185 | 180 | 185 | 265 | 185 | 85 | 5 |
| rapid | black | 310 | 295 | 315 | 380 | 315 | 85 | 20 |
| blitz | black | 405 | 385 | 415 | 445 | 395 | 60 | 30 |

## table_model_misfit

| alpha | band | model | games with P(win) < alpha | underdog wins: expected vs observed | observed / expected (approx. 95%) |
|---|---|---|---|---|---|
| 0.10 | <2200 | main | 90,081 | 5,948 vs 6,229 | 1.05 (1.02-1.07) |
| 0.10 | <2200 | interaction | 86,646 | 5,611 vs 5,841 | 1.04 (1.02-1.07) |
| 0.10 | 2200-2499 | main | 59,453 | 4,142 vs 3,769 | 0.91 (0.88-0.94) |
| 0.10 | 2200-2499 | interaction | 59,697 | 3,921 vs 3,816 | 0.97 (0.94-1.00) |
| 0.10 | 2500+ | main | 8,248 | 664 vs 512 | 0.77 (0.70-0.84) |
| 0.10 | 2500+ | interaction | 7,502 | 593 vs 461 | 0.78 (0.70-0.85) |
| 0.10 | all | main | 157,782 | 10,755 vs 10,510 | 0.98 (0.96-1.00) |
| 0.10 | all | interaction | 153,845 | 10,125 vs 10,118 | 1.00 (0.98-1.02) |
| 0.05 | <2200 | main | 23,911 | 875 vs 839 | 0.96 (0.89-1.02) |
| 0.05 | <2200 | interaction | 24,553 | 810 vs 837 | 1.03 (0.97-1.10) |
| 0.05 | 2200-2499 | main | 11,898 | 444 vs 308 | 0.69 (0.60-0.79) |
| 0.05 | 2200-2499 | interaction | 15,946 | 529 vs 491 | 0.93 (0.84-1.01) |
| 0.05 | 2500+ | main | 360 | 15 vs 14 | 0.92 (0.43-1.41) |
| 0.05 | 2500+ | interaction | 543 | 22 vs 23 | 1.06 (0.64-1.47) |
| 0.05 | all | main | 36,169 | 1,333 vs 1,161 | 0.87 (0.82-0.92) |
| 0.05 | all | interaction | 41,042 | 1,360 vs 1,351 | 0.99 (0.94-1.05) |

Model vs data in 50-point bins (n >= 100 games), observed - predicted:

| time control | band | model | bins inside the 95% interval | mean error, n-weighted (pp) |
|---|---|---|---|---|
| classical | <2200 | main | 34/38 | +0.20 |
| classical | <2200 | interaction | 34/38 | +0.22 |
| classical | 2200-2499 | main | 17/27 | +0.08 |
| classical | 2200-2499 | interaction | 22/27 | +0.06 |
| classical | 2500+ | main | 5/14 | -2.03 |
| classical | 2500+ | interaction | 5/14 | -2.16 |
| rapid | <2200 | main | 33/38 | +0.02 |
| rapid | <2200 | interaction | 35/38 | +0.04 |
| rapid | 2200-2499 | main | 25/26 | +0.39 |
| rapid | 2200-2499 | interaction | 25/26 | +0.27 |
| rapid | 2500+ | main | 9/12 | -0.88 |
| rapid | 2500+ | interaction | 11/12 | -0.87 |
| blitz | <2200 | main | 31/34 | +0.09 |
| blitz | <2200 | interaction | 33/34 | +0.11 |
| blitz | 2200-2499 | main | 26/26 | +0.27 |
| blitz | 2200-2499 | interaction | 26/26 | +0.29 |
| blitz | 2500+ | main | 13/14 | -0.63 |
| blitz | 2500+ | interaction | 12/14 | -0.74 |

Reading: in the tail (P(win) < 0.10) strong underdogs (2500+) win only 0.77 times the number the main model expects and 0.78 times with the interaction model, so adding the rating x gap terms does not fix the strong-player misfit; it does fix the 2200-2499 tail (0.91 -> 0.97). The intervals ignore tournament clustering and parameter uncertainty, so they are too narrow.

## table_alpha005_caveat

| alpha | underdog | band | time control | main (95% CI) | interaction (95% CI) | change | inside main CI |
|---|---|---|---|---|---|---|---|
| 0.20 | white | <2200 | classical | 145 (140-145) | 150 (145-150) | +5 | NO |
| 0.20 | white | <2200 | rapid | 230 (225-240) | 235 (225-245) | +5 | yes |
| 0.20 | white | <2200 | blitz | 285 (275-300) | 285 (275-300) | +0 | yes |
| 0.20 | white | 2200-2499 | classical | 100 (95-105) | 105 (100-110) | +5 | yes |
| 0.20 | white | 2200-2499 | rapid | 210 (200-220) | 220 (205-230) | +10 | yes |
| 0.20 | white | 2200-2499 | blitz | 270 (255-280) | 270 (255-285) | +0 | yes |
| 0.20 | white | 2500+ | classical | 55 (50-60) | 60 (50-65) | +5 | yes |
| 0.20 | white | 2500+ | rapid | 190 (180-205) | 205 (190-225) | +15 | yes |
| 0.20 | white | 2500+ | blitz | 250 (235-265) | 255 (235-275) | +5 | yes |
| 0.20 | black | <2200 | classical | 60 (55-65) | 60 (55-65) | +0 | yes |
| 0.20 | black | <2200 | rapid | 155 (150-165) | 160 (155-170) | +5 | yes |
| 0.20 | black | <2200 | blitz | 240 (225-250) | 235 (225-250) | -5 | yes |
| 0.20 | black | 2200-2499 | classical | 0 (0-0) | 0 (0-0) | +0 | yes |
| 0.20 | black | 2200-2499 | rapid | 125 (115-130) | 135 (125-145) | +10 | NO |
| 0.20 | black | 2200-2499 | blitz | 210 (200-225) | 215 (205-235) | +5 | yes |
| 0.20 | black | 2500+ | classical | 0 (0-0) | 0 (0-0) | +0 | yes |
| 0.20 | black | 2500+ | rapid | 90 (80-100) | 95 (75-115) | +5 | yes |
| 0.20 | black | 2500+ | blitz | 185 (170-200) | 195 (180-220) | +10 | yes |
| 0.10 | white | <2200 | classical | 320 (315-330) | 315 (310-320) | -5 | yes |
| 0.10 | white | <2200 | rapid | 425 (405-450) | 420 (400-440) | -5 | yes |
| 0.10 | white | <2200 | blitz | 485 (460-515) | 475 (450-510) | -10 | yes |
| 0.10 | white | 2200-2499 | classical | 280 (270-285) | 270 (265-280) | -10 | yes |
| 0.10 | white | 2200-2499 | rapid | 405 (385-425) | 395 (370-420) | -10 | yes |
| 0.10 | white | 2200-2499 | blitz | 465 (440-495) | 440 (415-470) | -25 | yes |
| 0.10 | white | 2500+ | classical | 235 (225-240) | 235 (225-240) | +0 | yes |
| 0.10 | white | 2500+ | rapid | 385 (365-405) | 370 (350-400) | -15 | yes |
| 0.10 | white | 2500+ | blitz | 450 (425-475) | 410 (385-450) | -40 | NO |
| 0.10 | black | <2200 | classical | 235 (235-240) | 235 (235-240) | +0 | yes |
| 0.10 | black | <2200 | rapid | 345 (335-360) | 345 (335-360) | +0 | yes |
| 0.10 | black | <2200 | blitz | 430 (405-455) | 430 (410-460) | +0 | yes |
| 0.10 | black | 2200-2499 | classical | 185 (180-185) | 185 (180-190) | +0 | yes |
| 0.10 | black | 2200-2499 | rapid | 310 (300-325) | 315 (300-335) | +5 | yes |
| 0.10 | black | 2200-2499 | blitz | 405 (380-430) | 395 (375-430) | -10 | yes |
| 0.10 | black | 2500+ | classical | 125 (120-130) | 135 (130-140) | +10 | NO |
| 0.10 | black | 2500+ | rapid | 280 (265-295) | 295 (275-315) | +15 | yes |
| 0.10 | black | 2500+ | blitz | 380 (355-405) | 370 (345-405) | -10 | yes |
| 0.05 | white | <2200 | classical | 495 (480-510) | 460 (445-470) | -35 | NO |
| 0.05 | white | <2200 | rapid | 660 (600-745) | 600 (565-650) | -60 | yes |
| 0.05 | white | <2200 | blitz | 700 (635-795) | 650 (605-726) | -50 | yes |
| 0.05 | white | 2200-2499 | classical | 445 (435-460) | 400 (390-410) | -45 | NO |
| 0.05 | white | 2200-2499 | rapid | 625 (575-695) | 540 (505-585) | -85 | NO |
| 0.05 | white | 2200-2499 | blitz | 675 (616-775) | 580 (540-645) | -95 | NO |
| 0.05 | white | 2500+ | classical | 400 (390-410) | 355 (345-370) | -45 | NO |
| 0.05 | white | 2500+ | rapid | 595 (550-650) | 500 (465-550) | -95 | NO |
| 0.05 | white | 2500+ | blitz | 650 (595-750) | 530 (495-595) | -120 | NO |
| 0.05 | black | <2200 | classical | 390 (380-400) | 380 (370-385) | -10 | yes |
| 0.05 | black | <2200 | rapid | 560 (520-620) | 510 (485-545) | -50 | NO |
| 0.05 | black | <2200 | blitz | 670 (600-745) | 635 (590-715) | -35 | yes |
| 0.05 | black | 2200-2499 | classical | 335 (330-345) | 320 (315-330) | -15 | NO |
| 0.05 | black | 2200-2499 | rapid | 510 (480-555) | 455 (430-485) | -55 | NO |
| 0.05 | black | 2200-2499 | blitz | 620 (570-699) | 560 (520-620) | -60 | NO |
| 0.05 | black | 2500+ | classical | 280 (275-290) | 270 (265-280) | -10 | NO |
| 0.05 | black | 2500+ | rapid | 470 (440-505) | 420 (395-450) | -50 | NO |
| 0.05 | black | 2500+ | blitz | 580 (535-650) | 515 (475-570) | -65 | NO |

Change interaction - main by alpha (all bands): alpha 0.20: -5 to +15 (mean +4), 16 of 18 inside the main 95% interval; alpha 0.10: -40 to +15 (mean -6), 16 of 18 inside the main 95% interval; alpha 0.05: -120 to -10 (mean -54), 4 of 18 inside the main 95% interval. Because the interaction model's tail is better calibrated at alpha 0.05 (see table_model_misfit), the main model's alpha 0.05 critical gaps are probably too large; quote them only with this caveat.

## table_expected_upsets

| time control | alpha | games with P(win) < alpha | expected upsets | observed upsets | observed / expected | wins: exp vs obs (z) | draws: exp vs obs |
|---|---|---|---|---|---|---|---|
| classical | 0.20 | 416,564 | 55,415 | 55,738 | 1.01 | 51,508 vs 51,580 (+0.3) | 3,907 vs 4,158 |
| classical | 0.10 | 129,541 | 9,152 | 9,007 | 0.98 | 8,780 vs 8,583 (-2.2) | 372 vs 424 |
| classical | 0.05 | 30,709 | 1,106 | 977 | 0.88 | 1,097 vs 958 (-4.3) | 9 vs 19 |
| rapid | 0.20 | 69,167 | 10,279 | 10,322 | 1.00 | 8,932 vs 8,953 (+0.2) | 1,347 vs 1,369 |
| rapid | 0.10 | 19,994 | 1,476 | 1,483 | 1.01 | 1,394 vs 1,375 (-0.5) | 82 vs 108 |
| rapid | 0.05 | 4,051 | 175 | 150 | 0.86 | 175 vs 150 (-1.9) | 0 vs 0 |
| blitz | 0.20 | 32,443 | 4,977 | 5,019 | 1.01 | 4,353 vs 4,395 (+0.7) | 624 vs 624 |
| blitz | 0.10 | 8,247 | 645 | 624 | 0.97 | 581 vs 552 (-1.3) | 64 vs 72 |
| blitz | 0.05 | 1,409 | 63 | 56 | 0.88 | 62 vs 53 (-1.1) | 2 vs 3 |
| all | 0.20 | 518,174 | 70,671 | 71,079 | 1.01 | 64,793 vs 64,928 (+0.6) | 5,878 vs 6,151 |
| all | 0.10 | 157,782 | 11,273 | 11,114 | 0.99 | 10,755 vs 10,510 (-2.5) | 519 vs 604 |
| all | 0.05 | 36,169 | 1,344 | 1,183 | 0.88 | 1,333 vs 1,161 (-4.8) | 11 vs 22 |

## table_expected_upsets(by_band=True)

| time control | band | alpha | games with P(win) < alpha | expected upsets | observed upsets | observed / expected | wins: exp vs obs (z) | draws: exp vs obs |
|---|---|---|---|---|---|---|---|---|
| classical | <2200 | 0.10 | 69,588 | 4,866 | 5,197 | 1.07 | 4,530 vs 4,792 (+4.0) | 336 vs 405 |
| classical | <2200 | 0.05 | 19,672 | 699 | 694 | 0.99 | 690 vs 675 (-0.6) | 9 vs 19 |
| classical | 2200-2499 | 0.10 | 52,073 | 3,652 | 3,332 | 0.91 | 3,616 vs 3,313 (-5.2) | 36 vs 19 |
| classical | 2200-2499 | 0.05 | 10,685 | 392 | 269 | 0.69 | 392 vs 269 (-6.3) | 0 vs 0 |
| classical | 2500+ | 0.10 | 7,880 | 635 | 478 | 0.75 | 635 vs 478 (-6.5) | 0 vs 0 |
| classical | 2500+ | 0.05 | 352 | 15 | 14 | 0.94 | 15 vs 14 (-0.2) | 0 vs 0 |
| rapid | <2200 | 0.10 | 14,824 | 1,105 | 1,136 | 1.03 | 1,024 vs 1,028 (+0.1) | 82 vs 108 |
| rapid | <2200 | 0.05 | 3,197 | 139 | 123 | 0.88 | 139 vs 123 (-1.4) | 0 vs 0 |
| rapid | 2200-2499 | 0.10 | 4,908 | 349 | 319 | 0.91 | 349 vs 319 (-1.7) | 0 vs 0 |
| rapid | 2200-2499 | 0.05 | 848 | 36 | 27 | 0.76 | 36 vs 27 (-1.5) | 0 vs 0 |
| rapid | 2500+ | 0.10 | 262 | 21 | 28 | 1.31 | 21 vs 28 (+1.5) | 0 vs 0 |
| rapid | 2500+ | 0.05 | 6 | 0 | 0 | 0.00 | 0 vs 0 (-0.5) | 0 vs 0 |
| blitz | <2200 | 0.10 | 5,669 | 446 | 474 | 1.06 | 395 vs 409 (+0.7) | 51 vs 65 |
| blitz | <2200 | 0.05 | 1,042 | 47 | 44 | 0.93 | 45 vs 41 (-0.7) | 2 vs 3 |
| blitz | 2200-2499 | 0.10 | 2,472 | 192 | 144 | 0.75 | 178 vs 137 (-3.2) | 14 vs 7 |
| blitz | 2200-2499 | 0.05 | 365 | 16 | 12 | 0.74 | 16 vs 12 (-1.0) | 0 vs 0 |
| blitz | 2500+ | 0.10 | 106 | 8 | 6 | 0.72 | 8 vs 6 (-0.8) | 0 vs 0 |
| blitz | 2500+ | 0.05 | 2 | 0 | 0 | 0.00 | 0 vs 0 (-0.3) | 0 vs 0 |
| all | <2200 | 0.10 | 90,081 | 6,417 | 6,807 | 1.06 | 5,948 vs 6,229 (+3.8) | 469 vs 578 |
| all | <2200 | 0.05 | 23,911 | 885 | 861 | 0.97 | 875 vs 839 (-1.2) | 11 vs 22 |
| all | 2200-2499 | 0.10 | 59,453 | 4,192 | 3,795 | 0.91 | 4,142 vs 3,769 (-6.0) | 49 vs 26 |
| all | 2200-2499 | 0.05 | 11,898 | 444 | 308 | 0.69 | 444 vs 308 (-6.6) | 0 vs 0 |
| all | 2500+ | 0.10 | 8,248 | 664 | 512 | 0.77 | 664 vs 512 (-6.2) | 0 vs 0 |
| all | 2500+ | 0.05 | 360 | 15 | 14 | 0.92 | 15 vs 14 (-0.3) | 0 vs 0 |

