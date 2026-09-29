# Security

This is a local-first application. The terminal binds to `127.0.0.1` only and
rejects requests with a non-loopback `Host` or a cross-site `Origin` before
any route runs (DNS-rebinding / CSRF guard in `terminal/server.py`). No
telemetry, no outbound calls on the demo path.

If you find a security issue, please use GitHub's private vulnerability
reporting on this repository rather than a public issue.
