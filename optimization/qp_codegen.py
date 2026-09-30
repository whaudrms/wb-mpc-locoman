"""Persistent C evaluation cache for CasADi QP data (not a compiled QP solver)."""
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

import casadi as ca


def source_key(source):
    # Symbolic Opti instance names occur in generated comments. They do not
    # change numerical code and should not force a rebuild for every instance.
    code = re.sub(rb'/\*.*?\*/', b'', source, flags=re.DOTALL)
    identity = f'{ca.__version__}:{platform.machine()}:cc:-O2:-fPIC'.encode()
    return hashlib.sha256(identity + code).hexdigest()


def compile_functions(functions, cache_dir=None):
    cache = Path(cache_dir or Path(__file__).resolve().parents[1]/'.deps/qp_codegen')
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-', dir=cache) as scratch:
        build = Path(scratch)
        generator = ca.CodeGenerator('qp_data.c')
        for function in functions:
            generator.add(function)
        generator.generate(str(build.resolve())+'/')
        source = build/'qp_data.c'
        key = source_key(source.read_bytes())
        target = cache/key
        target.mkdir(exist_ok=True)
        library = target/'qp_data.so'
        if not library.exists():
            print('Compiling QP data evaluation (first use; cached for subsequent runs)...', flush=True)
            try:
                subprocess.run(['cc', '-O2', '-fPIC', '-shared', str(source),
                                         '-o', str(build/'qp_data.so'), '-lm'],
                                        text=True, capture_output=True, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                detail = getattr(error, 'stderr', None) or str(error)
                raise RuntimeError('QP data compilation failed; install a C compiler or set '
                                   f'qp_compile_data=False. {detail[-3000:]}') from error
            os.replace(build/'qp_data.so', library)
        return [ca.external(f.name(), str(library.resolve())) for f in functions], str(library)
