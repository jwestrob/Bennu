MATCH (g:Gene {contig:$contig})
WHERE g.start >= $start AND g.end <= $end
RETURN g
ORDER BY g.start;

