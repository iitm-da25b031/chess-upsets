from tc_parse import parse_tc, tc_class
# (string, expected base seconds, expected increment seconds, expected class)
CASES = [
 ("90+30",5400,30,"classical"), ("3+2",180,2,"blitz"), ("60+30",3600,30,"classical"),
 ("15+10",900,10,"rapid"), ("10+5",600,5,"rapid"), ("1800+3",1800,3,"rapid"),
 ("45+10",2700,10,"rapid"), ("20+5",1200,5,"rapid"), ("10+3",600,3,"rapid"),
 ("25+10",1500,10,"rapid"), ("60+15",3600,15,"classical"), ("5400+30",5400,30,"classical"),
 ("30+30",1800,30,"classical"), ("900+10",900,10,"rapid"), ("10+10",600,10,"rapid"),
 ("5+3",300,3,"blitz"), ("2700+15",2700,15,"classical"), ("15+5",900,5,"rapid"), ("20+10",1200,10,"rapid"),
 ("90 minutes for the first 40 moves, followed by 30 minutes for the rest of the game with anincrement of 30 seconds per move starting from move 1",5400,30,"classical"),
 ("90min+30sec/move",5400,30,"classical"),
 ("90 minutes with 30 second increment from move 1",5400,30,"classical"),
 ("G90' + 30 per move",5400,30,"classical"),
 ("90 min + 30 sec",5400,30,"classical"),
 ("Standard: 60 minutes with 30 seconds increment from move 1",3600,30,"classical"),
 ("45m+10s",2700,10,"rapid"),
 ("90 minutes + 30 seconds per move",5400,30,"classical"),
 ("45 mininutes for the game and 10 seconds increment from move 1",2700,10,"rapid"),
 ("10 min + 5 sec/move Fischer",600,5,"rapid"),
 ("90min + 30 sg",5400,30,"classical"),
 ("90 min + 30 seconds permove from move number one",5400,30,"classical"),
 ("8'+ 3",480,3,"rapid"),
 ("90+30´ 30",5400,30,"classical"),
 ("90 min / 40 moves + 30 min + 30 sec / move",5400,30,"classical"),
 ("10 min + 5 seg add",600,5,"rapid"),
 ("G: 20 minutes + 5 seconds increment from move 1",1200,5,"rapid"),
 ("90min+30sec",5400,30,"classical"),
 ("90 min + 30 sec 40/5",5400,30,"classical"),
 ("90 min + 30 sg",5400,30,"classical"),
 # strings that failed or were mis-parsed on the full data (all 81 months)
 ("600",600,0,"blitz"), ("1500",1500,0,"rapid"), ("3600",3600,0,"classical"), ("10800",10800,0,"classical"),
 ("5400/40+30:1800+30\\\"",5400,30,"classical"), ("5400/40+30:1800",5400,30,"classical"),
 ("40/5400+30:1800",5400,30,"classical"), ("5400+30+1800",5400,30,"classical"),
 ("5400:30",5400,30,"classical"), ("900 + 5 sec",900,5,"rapid"), ("180+2",180,2,"blitz"),
]
bad = 0
for s, b, i, c in CASES:
    pb, pi, meth = parse_tc(s)
    pc = tc_class(pb, pi)
    ok = (pb, pi, pc) == (b, i, c)
    bad += not ok
    if not ok:
        print("MISMATCH", repr(s[:60]), "->", pb, pi, pc, meth, "expected", b, i, c)
print(f"{len(CASES)-bad}/{len(CASES)} correct")
print(parse_tc(None), parse_tc("   "), parse_tc("blah"))
