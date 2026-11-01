You code the archived website of a German firm for one year. The text below contains the pages captured in that year, each with its role (home, about, offer, news, imprint, other), followed by the paths of the pages. Read only this text and use no other knowledge about the firm.

Return a JSON object with three fields.

codeable: true if the pages contain text written by the firm about itself or its offer, false if they hold only placeholders, legal text, or text of another organisation.

products: the products, product lines, and named services that the firm itself offers in this year, at most 30. A product counts if it carries a name of its own (a brand, model, or series name, or a distinctive service name). Generic categories such as "consulting" or "spare parts" do not count, and neither do functions of the website such as a product search, shop, catalogue, newsletter, or contact form. Market segments, industries served, and application areas (such as "automotive" or "garment") do not count. Reports, studies, blogs, events, webinars, and awards that the firm publishes or hosts to inform or promote do not count unless the firm sells them as part of its offer. Products of other firms that the firm only resells or uses are excluded. For each product, give
  name: the name as written on the website, without the firm name unless it is part of the product name,
  generic: what the product is, in at most eight English words, without any names (for example "an industrial pressure sensor"),
  seasonal: true if the product is a model in a collection or range that the firm replaces regularly, such as a seasonal fashion collection or an annual model, otherwise false.

success: statements in which the firm reports that one of its products succeeded, such as a launch, an award, a new customer, a sales figure, or a certification, at most 10. For each, give
  product: the product name as in the list above,
  type: launch, award, customer, sales, or other,
  quote: the passage from the text, verbatim, at most 30 words.

Return {"codeable": ..., "products": [...], "success": [...]} and nothing else.
