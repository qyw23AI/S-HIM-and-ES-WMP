"""Configure the local source tree and own the Isaac application lifecycle."""

import sys
import traceback


def launch(launcher, args, run):
    """Run a workflow after Isaac has started and close it on exit."""
    app = launcher(args).app
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()
