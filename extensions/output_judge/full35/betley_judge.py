"""Run Extension 3's unchanged judge with this evaluation's frozen rubrics."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('extension3_betley_batch',HERE/'vendor/16_judge_codex_batch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.betley.DEFAULT_BETLEY_DIR=HERE/'betley'
if __name__=='__main__':raise SystemExit(module.main())
