# Security Policy

## Supported versions

CMF is pre-1.0. Security fixes are applied to the latest `0.x` release and the
`master` branch.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately using GitHub's
[private vulnerability reporting](https://github.com/infosave2007/cmf/security/advisories/new),
or email **urevich55@gmail.com** with the subject line `CMF SECURITY`.

Please include:

- a description of the issue and its impact;
- the affected component (`cortiq-core`, `cortiq-engine`, `cortiq-server`,
  `cortiq-cli`, or a converter) and version / commit;
- steps to reproduce, and a proof of concept if you have one.

We aim to acknowledge reports within **72 hours** and to provide a remediation
timeline after triage. We will credit reporters who wish to be named once a fix
is released.

## Threat model notes

CMF containers are parsed from untrusted input, so parsing robustness matters.
When reviewing or reporting, pay particular attention to:

- the container envelope and section-table parsing in `cortiq-core`
  (offsets, lengths, and bounds while memory-mapping);
- the delta index and per-skill replacement-tensor records;
- tokenizer and chat-template deserialization.

Treat any `.cmf` file from an untrusted source as untrusted data, the same way
you would treat any other model file.
