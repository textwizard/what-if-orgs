You code the archived website of a German firm for one year, for characteristics that describe its situation. The text below contains the pages captured in that year, each with its role, followed by the paths of the pages. Read only this text and use no other knowledge about the firm.

A signal counts only if the text gives concrete evidence about this firm itself, as current or as having happened within the past two years. Concrete evidence names an action, a place, a unit, a number, a date, a project, or a partner. The following do not count: general self-description and advertising language such as "we are innovative" or "we invest continuously", events described as part of the firm's history, menu labels and page titles alone, entries of a timeline or list that only name a year and a place, and products or services the firm sells to its customers, such as recruiting services of a staffing firm or research services for clients.

For each signal return a value, a strength, and a quote. The quote is a passage from the text, verbatim, at most 30 words, that shows the evidence. The strength is 3 if the evidence is specific and dated or quantified, 2 if it is specific, and 1 if it is plausible but vague. If there is no evidence, the value is false, the strength 0, and the quote null.

legal_name: the legal name of the firm as given in the imprint, or null.
employees: the number of employees the firm states, as a number, or null.
hiring: the firm advertises named open positions for itself or states that it is recruiting.
expansion: the firm reports a new location, plant, subsidiary, acquisition, entry into a new market, or enlarged capacity.
investment: the firm reports a named investment in facilities, equipment, information technology, or a new business field.
growth: the firm reports growth of its sales, staff, or customers, with a figure or a comparison.
difficulty: the firm reports losses, falling sales, short-time work, job cuts, restructuring, closures, or insolvency, including closures or declines due to the pandemic.
rd: the firm reports research and development of its own, such as a named development department or laboratory, development engineers, products it developed itself, or patents.
cooperation: the firm reports cooperation in research or development with named universities or research institutes, or a named publicly funded research project. Commercial partnerships with other firms do not count.

Return {"legal_name": ..., "employees": ..., "hiring": {"value": ..., "strength": ..., "quote": ...}, "expansion": {...}, "investment": {...}, "growth": {...}, "difficulty": {...}, "rd": {...}, "cooperation": {...}} and nothing else.
