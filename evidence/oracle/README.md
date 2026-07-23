# Oracle/Web ChatGPT advisory records

## Files

- [`2026-07-21_slip_pivot_audit_raw.md`](2026-07-21_slip_pivot_audit_raw.md): hostile
  journal-editor audit of the Ackermann/Pivot hypothesis.
- [`2026-07-21_journal_tier_context_raw.md`](2026-07-21_journal_tier_context_raw.md):
  preceding venue-tier context used by the follow-up conversation.

## Status

The slip/Pivot audit was requested through the Oracle browser workflow with `model=gpt-5-pro`
as a follow-up to an existing Pro-reviewed conversation. For the follow-up, the tool recorded:

```text
requested=Pro
resolved=(unavailable)
verified=no
```

The raw output footer labels the run `gpt-5-pro[browser]`, but the continuation did not provide
independent picker verification. The records are therefore advisory provenance, not proof of a
server-side model identity and not a primary scientific source.

The curated documents independently check the manual geometry, retain uncertainty, remove broad
claims contradicted by prior work, and cite peer-reviewed DOI records.

## Curated-value correction

The raw slip/Pivot advisory used the manual's isolated `0.324 m` wheelbase occurrence and therefore
reported approximately `39.72 deg` and its derived values. The complete-PDF count found six
occurrences of `0.323 m` against one occurrence of `0.324 m`. Curated documents and simulation
configuration consequently use `0.323 m` as the active default, recompute the ideal Pivot angle as
approximately `39.63 deg`, and preserve the raw advisory unchanged as provenance.
