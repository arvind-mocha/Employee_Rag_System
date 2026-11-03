"""Paid-call-free benchmark: PYTHONPATH=.:tests python tests/benchmark_evaluation_parallel.py."""
import json
from test_parallel_evaluation import Harness


def main():
    reports = []
    for count, concurrency in [(1,4),(4,4),(36,1),(36,4),(36,8)]:
        harness = Harness(delay=.05)
        run = harness.run(count,concurrency)
        report = dict(cases=count,concurrency=concurrency,**run.performance.model_dump())
        reports.append(report)
    baseline = next(r['elapsed_seconds'] for r in reports if r['cases']==36 and r['concurrency']==1)
    for r in reports:
        r['speedup_vs_sequential'] = round(baseline/r['elapsed_seconds'],2) if r['cases']==36 else None
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    main()
