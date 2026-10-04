"""Let the unchanged upstream adapter close Playwright on a managed stop."""
import signal


def stop(*_args):
    raise KeyboardInterrupt


def main():
    import server
    signal.signal(signal.SIGTERM, stop)
    try:
        server.main()
    except KeyboardInterrupt:
        # server.main's finally block closes its HTTP server and browser worker.
        pass


if __name__ == '__main__':
    main()
