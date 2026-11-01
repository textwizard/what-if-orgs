You code the archived website of a German firm for one year, for characteristics that describe its situation. The text below contains the pages captured in that year, each with its role, followed by the paths of the pages. Read only this text and use no other knowledge about the firm.

Return a JSON object with the following fields. Each signal is true only if the text states it for this firm as current or as having happened within the past two years, and is then supported by a passage from the text, verbatim, at most 30 words. Events described as part of the firm's history count as false. Otherwise the signal is false and the quote is null.

legal_name: the legal name of the firm as given in the imprint, or null.
employees: the number of employees the firm states, as a number, or null.
hiring: the firm advertises open positions or reports that it is hiring.
expansion: the firm reports new locations, plants, subsidiaries, or enlarged capacity.
investment: the firm reports investments in facilities, equipment, or new business fields.
difficulty: the firm reports losses, falling sales, job cuts, restructuring, or closures.
rd: the firm reports research and development of its own, such as a development department, laboratories, or engineers who develop its products.
cooperation: the firm reports cooperation in research or development with universities, research institutes, or publicly funded research projects.

Return {"legal_name": ..., "employees": ..., "hiring": {"value": ..., "quote": ...}, "expansion": {...}, "investment": {...}, "difficulty": {...}, "rd": {...}, "cooperation": {...}} and nothing else.
