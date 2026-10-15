from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


if __name__ == "__main__":
    
    rows = {}
    
    for path in sorted((RESULTS / "configs").glob("*.json")):
        
        c = json.loads(path.read_text(encoding='utf-8'))
        cid, cfg = c['config_id'], c['config']
        run = json.loads((RESULTS / "runs" / f"{cid}.json").read_text(encoding='utf-8'))
        analysis = json.loads((RESULTS / "analysis" / f"{cid}.json").read_text(encoding='utf-8'))
        coherence = analysis['coherence']['mean'] if analysis['coherence'] else None
        
        row = {'name': cfg['name'], 'config_id': cid, 'strategy': cfg['strategy'], 'size': cfg['size'],
               'overlap': cfg['overlap'], 'prefix': cfg['prefix'], 'n_chunks': c['n_chunks'],
               'truncation_rate': round(c['truncation_rate'], 4),
               'coherence_mean': None if coherence is None else round(coherence, 4)}
        
        for k, v in run['scores'].items():
            row[k] = v if (v is None or k == 'n') else round(v, 4)
        
        rows.setdefault(cfg['corpus'], []).append(row)
    
    for corpus, rs in rows.items():
        
        rs.sort(key=lambda r: r['name'])
        
        with open(RESULTS / f"summary_{corpus}.csv", 'w', newline='', encoding='utf-8') as f:
            
            writer = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            writer.writeheader()
            writer.writerows(rs)
        
        print(f"wrote summary_{corpus}.csv ({len(rs)} rows)")