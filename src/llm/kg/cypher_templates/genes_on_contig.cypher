MATCH (g:Gene {contig:$contig})
RETURN g
ORDER BY g.start;

