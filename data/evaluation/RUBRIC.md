# Answer-quality rubric

Review each generated answer in `reports/generated/answer_review.csv`.

- **Faithful:** Every material claim is entailed by the displayed evidence.
- **Citations correct:** Each cited source supports the associated claim and opens the stated filing.
- **Numeric accuracy:** Values, signs, units, and reporting periods match SEC XBRL or filing tables.
- **Refusal correct:** Unsupported forecasts, confidential information, and personalized advice are refused.

Use `1` for pass, `0` for fail, and `NA` only when a metric does not apply. Report the mean of
applicable rows and retain review notes for every failure.
