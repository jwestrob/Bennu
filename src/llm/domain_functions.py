#!/usr/bin/env python3
"""
Domain function annotation for biological context.
Maps protein domain names to functional descriptions.
"""

from typing import Dict, List, Optional
import re


# Known protein domain functions for biological context
DOMAIN_FUNCTIONS = {
    'GGDEF': {
        'function': 'Diguanylate cyclase activity',
        'description': 'Synthesizes c-di-GMP, a bacterial second messenger involved in biofilm formation and motility regulation',
        'pathway': 'Signal transduction',
        'importance': 'Critical for bacterial lifestyle switching between planktonic and biofilm states'
    },
    'PAS': {
        'function': 'Environmental sensing',
        'description': 'PAS domains bind small molecules and sense environmental conditions like oxygen, light, or metabolites',
        'pathway': 'Signal transduction',
        'importance': 'Allows bacteria to respond to environmental changes'
    },
    'PAS_3': {
        'function': 'Environmental sensing',
        'description': 'PAS domains bind small molecules and sense environmental conditions',
        'pathway': 'Signal transduction',
        'importance': 'Environmental response and regulation'
    },
    'PAS_4': {
        'function': 'Environmental sensing',
        'description': 'PAS domains involved in protein-protein interactions and signal transduction',
        'pathway': 'Signal transduction',
        'importance': 'Regulatory protein interactions'
    },
    'EAL': {
        'function': 'Phosphodiesterase activity',
        'description': 'Degrades c-di-GMP, antagonizing GGDEF domain function',
        'pathway': 'Signal transduction',
        'importance': 'Balances c-di-GMP levels for proper bacterial regulation'
    },
    'HD-GYP': {
        'function': 'Phosphodiesterase activity',
        'description': 'Alternative c-di-GMP degradation domain',
        'pathway': 'Signal transduction',
        'importance': 'c-di-GMP turnover and regulation'
    },
    'GAF': {
        'function': 'Ligand binding',
        'description': 'Binds cyclic nucleotides, heme, or other small molecules',
        'pathway': 'Signal transduction',
        'importance': 'Sensory input for regulatory systems'
    }
}


def extract_domains_from_ids(domain_ids: List[str]) -> List[str]:
    """Extract domain names from domain IDs."""
    domains = []
    for domain_id in domain_ids:
        if '/domain/' in domain_id:
            # Extract domain name from path like: .../domain/GGDEF/835-992
            match = re.search(r'/domain/([^/]+)/', domain_id)
            if match:
                domains.append(match.group(1))
    return list(set(domains))


def annotate_protein_domains(domain_ids: List[str]) -> Dict[str, any]:
    """Provide functional annotation for protein domains."""
    domain_names = extract_domains_from_ids(domain_ids)
    
    annotations = {
        'domain_names': domain_names,
        'domain_count': len(domain_ids),
        'unique_domain_types': len(domain_names),
        'functions': [],
        'pathways': set(),
        'biological_significance': []
    }
    
    for domain in domain_names:
        if domain in DOMAIN_FUNCTIONS:
            func_info = DOMAIN_FUNCTIONS[domain]
            annotations['functions'].append({
                'domain': domain,
                'function': func_info['function'],
                'description': func_info['description'],
                'pathway': func_info['pathway'],
                'importance': func_info['importance']
            })
            annotations['pathways'].add(func_info['pathway'])
    
    # Predict protein function based on domain combination
    if 'GGDEF' in domain_names and any('PAS' in d for d in domain_names):
        annotations['predicted_function'] = 'Environmental sensor and c-di-GMP signaling protein'
        annotations['biological_significance'].append(
            'This protein likely acts as an environmental sensor that regulates biofilm formation and bacterial motility in response to environmental cues'
        )
    elif 'GGDEF' in domain_names:
        annotations['predicted_function'] = 'Diguanylate cyclase signaling protein'
        annotations['biological_significance'].append(
            'Involved in c-di-GMP synthesis for biofilm regulation'
        )
    elif any('PAS' in d for d in domain_names):
        annotations['predicted_function'] = 'Environmental sensing protein'
        annotations['biological_significance'].append(
            'Senses environmental conditions and likely regulates cellular responses'
        )
    
    annotations['pathways'] = list(annotations['pathways'])
    
    return annotations


def format_domain_annotation(annotation: Dict[str, any]) -> str:
    """Format domain annotation for LLM context."""
    parts = []
    
    if annotation['domain_names']:
        parts.append(f"Domain Types: {', '.join(annotation['domain_names'])}")
    
    if annotation['functions']:
        parts.append("Domain Functions:")
        for func in annotation['functions']:
            parts.append(f"  • {func['domain']}: {func['function']} - {func['description']}")
    
    if annotation.get('predicted_function'):
        parts.append(f"Predicted Function: {annotation['predicted_function']}")
    
    if annotation['biological_significance']:
        parts.append("Biological Significance:")
        for sig in annotation['biological_significance']:
            parts.append(f"  • {sig}")
    
    if annotation['pathways']:
        parts.append(f"Pathways: {', '.join(annotation['pathways'])}")
    
    return '\n'.join(parts)