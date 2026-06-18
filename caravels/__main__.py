"""Entry point: python -m caravels [--dry-run] [--dashboard] [--with-dashboard]"""

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

from .config import AppConfig


def _setup_logging(cfg: AppConfig, log_level: str | None, log_file: bool) -> None:
    level = getattr(logging, (log_level or cfg.log_level).upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(cfg.db_path).parent / "caravels.log"
        fh = logging.handlers.TimedRotatingFileHandler(log_path, when="midnight", backupCount=14, utc=True)
        fh.setFormatter(fmt)
        handlers.append(fh)
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Caravels — autonomous trading vessel")
    parser.add_argument("--dry-run", action="store_true", help="Paper mode — no real trades")
    parser.add_argument("--dashboard", action="store_true", help="Launch operator dashboard only (no agent loop)")
    parser.add_argument("--with-dashboard", action="store_true", help="Run agent loop AND dashboard concurrently")
    parser.add_argument("--settings", default=None, metavar="FILE", help="Path to settings JSON (default: settings.json)")
    parser.add_argument("--log-level", default=None, help="Override log level (DEBUG/INFO/WARNING)")
    parser.add_argument("--no-log-file", action="store_true", help="Disable rotating log file (console only)")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port (default 5050)")
    args = parser.parse_args()

    cfg = AppConfig.from_env(settings_path=args.settings)
    if args.dry_run:
        cfg = cfg.with_dry_run(True)

    _setup_logging(cfg, args.log_level, log_file=not args.no_log_file)
    logging.getLogger(__name__).info(
        "Settings loaded: profile=%s strategy=%s min_trade=$%.2f",
        "custom" if args.settings else "default",
        cfg.strategy_version,
        cfg.risk.min_trade_notional_usd,
    )

    if args.dashboard:
        # Dashboard-only mode (no agent loop)
        import waitress

        from .app import create_app

        app = create_app(cfg)
        logging.getLogger(__name__).info("Dashboard starting on http://0.0.0.0:%d", args.port)
        try:
            waitress.serve(app, host="0.0.0.0", port=args.port)
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Dashboard stopped by user")
        return

    if args.with_dashboard:
        # Start dashboard in background thread, then run the agent loop
        import threading

        import waitress

        from .app import create_app

        app = create_app(cfg)
        t = threading.Thread(
            target=waitress.serve,
            kwargs={"app": app, "host": "0.0.0.0", "port": args.port},
            daemon=True,
            name="caravels-dashboard",
        )
        t.start()
        logging.getLogger(__name__).info("Dashboard started on http://0.0.0.0:%d", args.port)

    from .run import run_loop

    run_loop(cfg)


if __name__ == "__main__":
    main()
