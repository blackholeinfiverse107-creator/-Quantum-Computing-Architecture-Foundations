@echo off
echo Starting verification... > verification_log.txt
python --version >> verification_log.txt 2>&1
if exist run_tests.py (
    python run_tests.py >> verification_log.txt 2>&1
) else (
    echo run_tests.py not found >> verification_log.txt
)
echo Done. >> verification_log.txt
