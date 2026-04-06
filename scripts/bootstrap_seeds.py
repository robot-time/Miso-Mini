#!/usr/bin/env python3
"""Write data/seeds_bulk.jsonl — diverse questions + some long-context rows for compression."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "seeds_bulk.jsonl"

QUESTIONS = [
    # ML / stats
    ("What is the bias–variance tradeoff?", ""),
    ("When would you use cross-validation?", ""),
    ("What is regularization and why use it?", ""),
    ("Explain gradient descent in one tight paragraph.", ""),
    ("What is a confusion matrix?", ""),
    ("What does it mean for a model to be calibrated?", ""),
    ("What is transfer learning?", ""),
    ("What is an embedding in ML?", ""),
    # CS / systems
    ("What is Big-O notation and why does it matter?", ""),
    ("Explain TCP vs UDP in practical terms.", ""),
    ("What is a race condition?", ""),
    ("What is idempotency in APIs?", ""),
    ("What is eventual consistency?", ""),
    ("What is a deadlock?", ""),
    ("What is the difference between process and thread?", ""),
    ("What is virtual memory?", ""),
    ("What is a REST API?", ""),
    ("What is OAuth for, at a high level?", ""),
    ("What is a CDN?", ""),
    ("What is garbage collection?", ""),
    # Software / coding concepts
    ("What is dependency injection?", ""),
    ("What is a pure function?", ""),
    ("What is immutability and a benefit?", ""),
    ("What is a unit test vs integration test?", ""),
    ("What is technical debt?", ""),
    ("What is semantic versioning?", ""),
    ("What is CI/CD?", ""),
    ("What is a monorepo?", ""),
    ("What is an ORM?", ""),
    ("What is middleware in web apps?", ""),
    ("What is rate limiting?", ""),
    # Math / logic
    ("What is a prime number?", ""),
    ("What is modular arithmetic?", ""),
    ("What is Bayes' theorem in plain language?", ""),
    ("What is a derivative (intuitive)?", ""),
    ("What is a vector vs scalar?", ""),
    ("What is a matrix multiply used for in ML?", ""),
    # Product / communication
    ("How do you write a clear bug report?", ""),
    ("What is a user story?", ""),
    ("What is scope creep?", ""),
    ("How do you prioritize a backlog?", ""),
    ("What is an MVP?", ""),
    # Security
    ("What is SQL injection?", ""),
    ("What is XSS?", ""),
    ("What is MFA?", ""),
    ("What is the principle of least privilege?", ""),
    # Science / general reasoning
    ("Why is the sky blue (short)?", ""),
    ("What is photosynthesis in one sentence?", ""),
    ("What is evolution by natural selection?", ""),
    ("What is confirmation bias?", ""),
    ("What is Occam's razor?", ""),
    # Writing / analysis
    ("How do you summarize a long article?", ""),
    ("What makes an argument strong vs weak?", ""),
    ("What is a straw man fallacy?", ""),
    # Data / SQL
    ("What is a primary key?", ""),
    ("What is a JOIN in SQL?", ""),
    ("What is normalization in databases?", ""),
    ("What is an index in a database?", ""),
    # Misc practical
    ("How do you debug a failing script?", ""),
    ("What is latency vs throughput?", ""),
    ("What is caching?", ""),
    ("What is compression?", ""),
    ("What is lossy vs lossless?", ""),
    ("What is Unicode?", ""),
    ("What is UTF-8?", ""),
    ("What is JSON?", ""),
    ("What is YAML used for?", ""),
    ("What is Docker?", ""),
    ("What is Kubernetes at a high level?", ""),
    ("What is a load balancer?", ""),
    ("What is horizontal scaling?", ""),
    ("What is a message queue?", ""),
    ("What is idempotency in payments?", ""),
    ("What is double-entry bookkeeping?", ""),
    ("What is inflation?", ""),
    ("What is compound interest?", ""),
    ("What is opportunity cost?", ""),
    ("What is a Pareto principle use case?", ""),
]

LONG_FACTS = """
Project Orion is an internal initiative to consolidate three legacy billing systems into one API.
Stakeholders: Finance (accuracy), Support (refunds within 48h), and Compliance (audit logs 7 years).
Constraints: no downtime during US business hours; migration must be reversible for 30 days.
Risks: duplicate charges if webhooks retry; partial outages in region eu-west-1 last quarter.
Success metrics: p95 checkout latency under 400ms; dispute rate below 0.3%; zero P1 incidents in rollout.
Timeline: design freeze March 15; pilot 10% traffic April 1; full cutover May 1.
Open decisions: whether to store card tokens in-house or use vault provider; SLA for webhook delivery.
"""

LONG_NOTES = """
Meeting notes — Q2 planning. Alice proposed hiring two SREs; Bob worried about budget.
We agreed to pilot on-call rotation with existing team first. Action: Carol drafts runbook by Friday.
Customer churn spiked in segment SMB after price change; marketing to run retention experiment.
Engineering velocity down 15% due to incident load; propose no new features for two sprints.
Security review flagged weak MFA adoption; require MFA for admin roles next month.
"""

LONG_RESEARCH = """
Study summary (fictional): n=240 adults, 12-week intervention combining sleep hygiene coaching
and light exercise. Primary outcome: self-reported sleep quality (Likert). Secondary: HRV wearable data.
Results: intervention arm +0.8 SD vs control on sleep quality; adherence 68%; dropout 12%.
Limitations: self-report bias; no polysomnography; population mostly urban, ages 25–45.
Conclusion: promising but needs objective sleep measures and broader demographics.
"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"question": q, "context": c} for q, c in QUESTIONS]

    compression_qs = [
        "What are the main risks and constraints we should track?",
        "What decisions are still open?",
        "Summarize success metrics and timeline.",
        "What should engineering prioritize based on these notes?",
        "What are the study limitations and what would you do next?",
    ]
    blobs = [LONG_FACTS.strip(), LONG_NOTES.strip(), LONG_RESEARCH.strip()]
    for i, cq in enumerate(compression_qs):
        rows.append({"question": cq, "context": blobs[i % len(blobs)]})

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} seeds -> {OUT}")


if __name__ == "__main__":
    main()
