### Future: Final Single-Pass “Publish” Call

After incremental synthesis converges, we plan a final API call where gpt-5-high (or a cost-efficient alternative, e.g., gpt-5-medium or 4.1-mini for templated prose) renders the polished final report from the IRB AST. This step is not implemented now; it should be configurable and budget-aware. Keep the IRB validators and claims ledger as the source of truth; the publish step must not introduce new claims—formatting and phrasing only.

