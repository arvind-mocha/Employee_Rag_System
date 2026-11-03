# TCS - Technology Consulting Sandbox | KRAG Test Dataset

**This corpus is 100% synthetic.** It is unrelated to real Tata Consultancy Services. Do not present any of its company, people, customer, profit, tax or policy details as real-world facts. The email domains use the reserved `.example` namespace and are not deliverable.

## Snapshot and coverage
- Snapshot: 2026-11-02
- Employees: 80; departments: 8; one manager / one team lead / two seniors / three mid-seniors / two associates / one intern per department.
- Locations: London, Berlin, Paris, Amsterdam, New York, Austin, Toronto, São Paulo.
- Clients: 8 fictional B2B accounts.
- History: FY2024 and FY2025 finance; Jan-Sep 2026 management accounts; superseded 2025 handbook and current 2026 handbook.
- Format mix: PDF, DOCX, CSV, JSON, Markdown, TXT and XLSX (where supported by your KRAG ingestion pipeline).

## Which folders should be ingested?
**Ingest only `corpus/`.** Do not ingest `evaluation_do_not_ingest/` or `metadata/` when grading, because they contain the reference answers and dataset notes. The evaluation file deliberately lives outside the corpus to avoid answer leakage.

## Structure
- `corpus/00_company/`: Business overview and FAQ.
- `corpus/01_policies/`: HR, security, delivery SLAs, governance, procurement, and a clearly archived 2025 policy.
- `corpus/02_employees/`: CSV/JSON/XLSX directory, 80 individual profiles, eight department rosters, management hierarchy.
- `corpus/03_clients_services/`: Seven-service catalog, eight client account briefs, eight project briefs.
- `corpus/04_finance/`: Annual and quarterly finance, tax allocation, client and region revenue, CSV extracts, XLSX workbook.
- `corpus/05_emails/`: 15 synthetic intra-company email messages in plain text.
- `evaluation_do_not_ingest/krag_eval_questions_45.jsonl`: 45 test questions with `question`, `ground_truth`, `expected_answer`, `source_documents`, `category` and `difficulty`.
- `metadata/corpus_manifest.csv`: Hash manifest for ingestion audit.

## Test use cases
1. Exact-match company and employee fact lookup.
2. Multi-hop reporting chains / client owner to line manager.
3. Year-aware answers: archived 2025 remote policy vs current 2026 version.
4. Arithmetic reconciliation (client revenue total, regional split, tax provision).
5. Tabular extraction from spreadsheets and CSV; semantic extraction from DOCX and PDF.
6. Email reference retrieval vs policy source of truth.
7. Grounded answers and abstentions: the system must not fabricate unlisted contract details or employees.

## Canonical figures
- FY2025 revenue: USD 18.4m; operating expenses: USD 14.9m; operating profit: USD 3.5m; tax provision: USD 0.875m; net profit: USD 2.625m.
- FY2025 client region split: Europe USD 9.4m, Americas USD 9.0m.
- 2026 Q3 YTD: revenue USD 17.0m, op. expense USD 13.5m, op. profit USD 3.5m, tax USD 0.875m, net USD 2.625m.
- Grade rules are stated in `HR_Role_Grades_and_Reporting_v2_2026.docx`.

**Note on JSONL:** The file includes both `ground_truth` and `expected_answer` for convenience. If your evaluator expects a different field name, map it on import. Do not index ground truth as document context.
