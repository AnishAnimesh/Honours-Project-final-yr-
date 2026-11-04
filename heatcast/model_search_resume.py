"""Transfer only provably compatible OOF caches after a bounded-search amendment."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from . import model_search


def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def transfer(source, destination, data='data/europe_sites/daily.npz'):
    source, destination = Path(source), Path(destination)
    old = json.loads((source/'manifest.json').read_text())
    current_path = Path(model_search.__file__)
    snapshot = source/'model_search_source.py'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == old['hashes'][str(current_path)]
    before, after = functions(snapshot.read_text()), functions(current_path.read_text())
    # The revision changes only the candidate generator, not feature/label/fold logic.
    assert before.keys() == after.keys()
    for name in before.keys()-{'candidate_specs'}:
        assert before[name] == after[name], f'Function changed: {name}'
    for path, expected in old['hashes'].items():
        if path not in [str(current_path), 'MODEL_SEARCH_PROTOCOL.md', 'data']:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256(Path(data).read_bytes()).hexdigest() == old['hashes']['data']
    versions = {p: importlib.metadata.version(p) for p in old['versions']}
    assert versions == old['versions'], 'Package versions differ'
    specs = model_search.candidate_specs()
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in old['hashes'] if p != 'data'}
    hashes['data'] = old['hashes']['data']
    identity = dict(hashes=hashes, candidates=specs, versions=versions,
                    leads=old['leads'], tasks=old['tasks'])
    identity['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        raise ValueError('Destination must be new')
    (destination/'cache').mkdir(parents=True)
    old_specs = {s['name']: s for s in old['candidates']}
    reused = []
    for spec in specs:
        if old_specs.get(spec['name']) != spec:
            continue
        for lead in old['leads']:
            for task in old['tasks']:
                name = f'{task}_lead{lead}_{spec["name"]}_oof.npz'
                if (source/'cache'/name).exists():
                    shutil.copy2(source/'cache'/name, destination/'cache'/name)
                    reused.append(name)
    (destination/'manifest.json').write_text(json.dumps(identity, indent=2))
    record = dict(source=str(source), source_fingerprint=old['fingerprint'],
                  destination_fingerprint=identity['fingerprint'], files=reused,
                  verification='All functions except candidate_specs AST-identical; '
                               'data, dependencies, package versions and candidate specs identical')
    (destination/'CACHE_REUSE.json').write_text(json.dumps(record, indent=2))
    print(f'Reused {len(reused)} verified candidate evaluations; changed candidates will be fitted afresh.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/model_search_initial')
    parser.add_argument('--destination', default='results/model_search')
    args = parser.parse_args()
    transfer(args.source, args.destination)


if __name__ == '__main__':
    main()
