from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Evaluate pipeline recommendations against deterministic ground truth labels.')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--out-dir', default='outputs')
    args = parser.parse_args()

    predictions = {r['provider_id']: r for r in read_jsonl(Path(args.predictions))}
    truth_rows = read_jsonl(Path(args.ground_truth))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    action_correct = 0
    false_auto_update = 0
    auto_updates = 0
    risk_expected = 0
    risk_caught = 0
    by_scenario = defaultdict(lambda: Counter())
    rows = []

    for gt in truth_rows:
        pid = gt['provider_id']
        pred = predictions.get(pid, {})
        expected = gt['expected_action']
        actual = pred.get('recommended_action', 'missing_prediction')
        total += 1
        if actual == expected:
            action_correct += 1
        if actual == 'auto_update':
            auto_updates += 1
        if actual == 'auto_update' and expected != 'auto_update':
            false_auto_update += 1
        if expected in {'human_review', 'outreach_required'}:
            risk_expected += 1
            if actual in {'human_review', 'outreach_required'}:
                risk_caught += 1
        by_scenario[gt.get('scenario', 'unknown')][actual] += 1
        rows.append({
            'provider_id': pid,
            'scenario': gt.get('scenario', ''),
            'expected_action': expected,
            'actual_action': actual,
            'correct': actual == expected,
            'num_changes': len(pred.get('changes', [])) if pred else 0,
            'reason': pred.get('reason', ''),
        })

    report = {
        'total': total,
        'action_accuracy': round(action_correct / total, 4) if total else 0,
        'auto_update_recommendations': auto_updates,
        'false_auto_update': false_auto_update,
        'false_auto_update_rate_among_auto_updates': round(false_auto_update / auto_updates, 4) if auto_updates else 0,
        'risk_case_recall': round(risk_caught / risk_expected, 4) if risk_expected else 0,
        'predicted_action_counts': Counter(p.get('recommended_action', 'missing_prediction') for p in predictions.values()),
        'scenario_breakdown': {k: dict(v) for k, v in by_scenario.items()},
    }

    (out / 'evaluation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'evaluation_rows.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['provider_id','scenario','expected_action','actual_action','correct','num_changes','reason'])
        writer.writeheader()
        writer.writerows(rows)

    md = [
        '# Evaluation Report',
        '',
        f"- Total evaluated: {report['total']}",
        f"- Action accuracy: {report['action_accuracy']}",
        f"- Auto-update recommendations: {report['auto_update_recommendations']}",
        f"- False auto-update: {report['false_auto_update']}",
        f"- False auto-update rate among auto-updates: {report['false_auto_update_rate_among_auto_updates']}",
        f"- Risk-case recall: {report['risk_case_recall']}",
        '',
        '## Scenario breakdown',
    ]
    for scenario, counts in report['scenario_breakdown'].items():
        md.append(f"- {scenario}: {counts}")
    (out / 'evaluation_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
