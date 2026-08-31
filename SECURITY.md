# Security Policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting flow from the repository's Security tab. Do not disclose an unpatched vulnerability in a public issue.

Include the affected revision, impact, reproduction steps, and any relevant environment details. Do not include production credentials, access tokens, proprietary workloads, customer data, or unrelated repository files.

## Workload privacy

The local package and composite GitHub Action execute in the caller's environment. The action does not provision hardware, upload source code, or send reports to an Accel-Verify service. Any artifact upload must be configured explicitly by the caller's workflow.
