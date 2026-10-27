&#x20;  # Chess upsets over the board (DA2402)

&#x20;  How large a rating gap does it take for an upset, and does it change with the time control?

&#x20;  Data: Lichess broadcast PGN archive + FIDE monthly rating lists.



&#x20;  ## Run order

&#x20;  1. python download\_lichess.py urls\_lichess.txt

&#x20;  2. python download\_fide.py

&#x20;  3. python run\_all.py

&#x20;  4. python combine.py

&#x20;  5. python clock\_features.py

&#x20;  6. python clean\_games.py

&#x20;  7. python diag\_elo.py

&#x20;  8. python missingness.py

&#x20;  9. python features.py



&#x20;  Data goes in data/ (not in git). Results of our runs are in reports/.

