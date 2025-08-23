MATCH (g:Gene {contig:$contig})
RETURN g
ORDER BY toInteger(g.startCoordinate);
