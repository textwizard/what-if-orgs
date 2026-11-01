You code the archived website of a German firm for one year. The text below contains the pages captured in that year, each with its role, followed by the paths of the pages. Read only this text and use no other knowledge about the firm.

Return a JSON object with three fields.

codeable: true if the pages contain text written by the firm about itself or its offer, false if they hold only placeholders, legal text, or text of another organisation.

products: the products, product lines, and named services that the firm itself offers in this year, at most 30. A product counts if it carries a name of its own (a brand, model, or series name, or a distinctive service name). Generic categories such as "consulting" or "spare parts" do not count, and neither do functions of the website such as a product search, shop, catalogue, newsletter, or contact form. Market segments, industries served, and application areas (such as "automotive" or "garment") do not count. Reports, studies, blogs, events, webinars, and awards that the firm publishes or hosts to inform or promote do not count unless the firm sells them as part of its offer. Products of other firms that the firm only resells or uses are excluded. For each product, give
  name: the name as written on the website, without the firm name unless it is part of the product name,
  generic: what the product is, in at most eight English words, without any names, in terms that another firm of the industry could use for a similar product (for example "an industrial pressure sensor"),
  seasonal: true if the product is a model in a collection or range that the firm replaces regularly, such as a seasonal fashion collection or an annual model, or a special, limited, or commemorative edition issued for an occasion, otherwise false. Give the same answer for the same kind of product in every year,
  stage: "pilot" if the text presents the product as a pilot, beta, prototype, or test, otherwise "market",
  scale: "line" if the product is a family of several models or variants, otherwise "single",
  featured: true if the text presents the product as a focus of the firm, such as a flagship, a new strategic field, or a product named in the firm's own description of its direction, otherwise false.

evidence: statements about one of the products above in this year, at most 15. Each statement must name the product, be specific to it, and refer to this year or the year before. General praise, advertising language, and descriptions of what the product does do not count. For each statement, give
  product: the product name as in the list above,
  type: one of
    "result" for evidence of how the product performs on the market, such as sales or units sold, named customers or orders, adoption by users, quantified figures of installations, users, or output achieved with the product, an award or test win, or a new version or new market that the firm attributes to its success. A launch alone is not a result, and neither is a certification, a test the product passed, a trade fair appearance, or a statement about the competence of the firm,
    "problem" for evidence that the product performs poorly, such as falling sales, a recall, a delay, or a discontinuation,
    "launch" for the announcement that the product is new or now available,
    "investment" for an investment dedicated to the product, such as a new plant, production line, team, or advertising campaign. Price promotions, discounts, contests, and trade fair appearances do not count,
  strength: 3 if the statement is quantified or dated, 2 if it is specific, 1 if it is plausible but vague,
  quote: the passage from the text, verbatim, at most 30 words.

Return {"codeable": ..., "products": [...], "evidence": [...]} and nothing else.
