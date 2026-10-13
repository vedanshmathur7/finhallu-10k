.PHONY: test questions filings run report agree-sample agree-score
test:
	python -m pytest -q && python -m finhallu.run --dry-run && python -m finhallu.report results/results_dryrun.csv > /dev/null
questions:
	python -m finhallu.questions
filings:
	python -m finhallu.corpus
run:
	python -m finhallu.run
smoke:
	python -m finhallu.run --limit 10
report:
	python -m finhallu.report
agree-sample:
	python scripts/agreement.py sample
agree-score:
	python scripts/agreement.py score
