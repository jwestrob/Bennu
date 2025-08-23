MATCH (g:Gene {contig:$contig})
WHERE toInteger(g.startCoordinate) >= $start AND toInteger(g.endCoordinate) <= $end
RETURN g
ORDER BY toInteger(g.startCoordinate);
