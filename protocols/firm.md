You describe a German firm from its archived website in one year. The text below contains the pages captured in that year, each with its role, followed by the paths of the pages. Read only this text and use no other knowledge about the firm.

Return a JSON object with two fields.

strategy: the account the firm gives of itself, with five fields of at most 25 English words each,
  offer: what the firm offers,
  customers: to whom,
  why_we_win: why the firm expects to succeed, in its own reasoning,
  resources: the resources and capabilities it names as its basis,
  direction: where it says it is heading, or null if it says nothing about this.
Write the fields as a paraphrase at a level of abstraction that other firms in the same industry could share. Leave out names of the firm, its products, brands, people, partners, and places, and leave out niche specialties, awards, claims of market leadership, founding stories, and any other detail from which a reader could identify the firm.

context: characteristics of the firm, each null if the text does not state it,
  size: micro (under 10 employees), small (10 to 49), medium (50 to 249), or large (250 or more), only from a stated number of employees, otherwise null,
  founded: the founding year, only if stated.

Return {"strategy": {...}, "context": {...}} and nothing else.
