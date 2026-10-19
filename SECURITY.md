# Security policy

FacePay is a research prototype for face + PIN payment authentication. It moves no money and is
not certified for production use. The design, threat model and known limitations are in
[docs/report.md](docs/report.md) (section 5).

## Supported code

| Path | Supported |
|---|---|
| `facepay/` (2.x) | ✅ security fixes |
| `legacy/` (the 2019 prototype) | ❌ archived baseline, known-vulnerable dependencies; never deploy it |

## Known limitations (by design, documented)

- The liveness check stops photos and screens that are tilted, moved or bent. It does **not**
  stop a replayed video of the account holder turning the requested way, real-time deepfakes or
  3D masks.
- Thresholds are calibrated on LFW, not on webcam captures of real users.
- Challenges live in process memory, so run one worker per instance (or move them to a shared store).

Please don't report these as vulnerabilities. Reports of ways around the controls in the
report's threat model are welcome.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting ("Report a vulnerability" on the Security tab).
Don't open a public issue. Expect an acknowledgement within a week.

## Deployment requirements

- Set `FACEPAY_SECRET_KEY` (a Fernet key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
  Without it a development key is generated next to the database, which defeats encryption at rest.
- Serve over HTTPS. Browsers only allow camera access on HTTPS or `localhost`.
- Keep `FACEPAY_ADMIN_TOKEN` unset unless you need the audit endpoint.
