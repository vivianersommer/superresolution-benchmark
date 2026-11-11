"""Refaz o experimento completo (fases 1 a 5) a partir das imagens.

Uso (a partir da raiz do repositório):
    .venv/bin/python run_all.py            # fases 1 a 5
    .venv/bin/python run_all.py --tests    # + pytest no fim
    .venv/bin/python run_all.py --from 3   # retoma a partir da fase 3

Cada etapa roda num subprocesso com o mesmo interpretador. Qualquer falha interrompe a
execução (sem valores sentinela). O log fica em outputs/run_all.log.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    (1, "src/f1_amostragem.py", "Fase 1 — inventário, amostra, HR com mod-crop, benchmarks"),
    (2, "src/f2_reconstrucao.py", "Fase 2 — LR (imresize MATLAB) e 1.400 reconstruções"),
    (3, "src/f3_metricas.py", "Fase 3 — validação Set5/Set14 e métricas"),
    (3, "src/f3_diagnostico.py", "Fase 3 — diagnóstico do déficit dos modelos"),
    (4, "src/f4_descritiva.py", "Fase 4 — estatística descritiva e figuras"),
    (5, "src/f5_inferencia.py", "Fase 5 — inferência (H1, H2, H3, poder)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="primeira fase a executar (1-5)")
    ap.add_argument("--tests", action="store_true", help="roda a suíte pytest ao final")
    args = ap.parse_args()

    log_path = ROOT / "outputs" / "run_all.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    total0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:
        def emit(msg: str) -> None:
            line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit(f"run_all.py — Python {sys.version.split()[0]} ({sys.executable})")
        for phase, script, desc in STEPS:
            if phase < args.start:
                continue
            emit(f"INÍCIO {desc} [{script}]")
            t0 = time.perf_counter()
            res = subprocess.run([sys.executable, script], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            dt = time.perf_counter() - t0
            if res.returncode != 0:
                emit(f"FALHA {script} (código {res.returncode}, {dt:.1f} s). Veja {log_path}.")
                return res.returncode
            emit(f"OK {script} ({dt:.1f} s)")
        if args.tests:
            emit("INÍCIO pytest")
            res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-W", "ignore", "tests/"],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if res.returncode != 0:
                emit(f"FALHA pytest (código {res.returncode}). Veja {log_path}.")
                return res.returncode
            emit("OK pytest")
        emit(f"CONCLUÍDO em {time.perf_counter() - total0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
