import os
import dspy
from .config import config
from .program import PromptXpertProgram
from .lm_init import init_lms


def load_program(path: str | None = None, best: bool = True) -> PromptXpertProgram:
    if path is None:
        # Use best program if available
        if best:
            candidate = os.path.join(config.artifacts_dir, 'best_program.json')
            if os.path.exists(candidate):
                path = candidate
            else:
                path = config.save_path
        else:
            path = config.save_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Compiled program artifact not found at {path}. Run optimization first or specify a valid path.")
    # Ensure an LM is configured (dspy requires a global lm setting)
    try:
        if getattr(dspy.settings, 'lm', None) is None:
            init_lms()
    except Exception:
        # Fallback: attempt init anyway
        init_lms()
    # If path is a directory, treat it as whole-program saved directory
    if os.path.isdir(path):
        return dspy.load(path if path.endswith('/') else path + '/')
    # If pointer file (JSON) referencing whole-program dir
    try:
        with open(path) as pf:
            maybe_pointer = pf.read(256)
        if 'redirect_to_whole_program_dir' in maybe_pointer:
            import json as _json
            data = _json.loads(open(path).read())
            target = data.get('redirect_to_whole_program_dir')
            if target and os.path.isdir(target):
                return dspy.load(target if target.endswith('/') else target + '/')
    except Exception:
        pass
    prog = PromptXpertProgram()
    prog.load(path)
    return prog


def optimize_prompt(prompt: str, program: PromptXpertProgram | None = None, path: str | None = None, best: bool = True) -> str:
    if program is None:
        program = load_program(path=path, best=best)
    pred = program(initial_prompt=prompt)
    return pred.optimized_prompt
