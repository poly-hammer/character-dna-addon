# Repository agent instructions

**Documentation framework:** Fumadocs. Read `.github/instructions/documentation.instructions.md` before editing `docs/`; the central docs site provides preview, Markdown, and MCP access.

## Documentation: Fumadocs and agent access

Documentation in `docs/` is authored for the central Poly Hammer Fumadocs site. Do not add MkDocs syntax, plugins, navigation, or commands.

Write the brand as **Poly Hammer** in visible text and metadata. Preserve existing URLs and code identifiers.

- Use Markdown/MDX with YAML `title` and `description` frontmatter. The layout renders the H1; start body sections at H2.
- Use `docs/meta.json` and folder `meta.json` files for navigation order and labels.
- Use `<Callout type="info|warn|error|idea|success" title="...">` instead of `!!!` admonitions. Use shared `DocVideo`, `DocImage`, and `YouTubeEmbed` components for media.
- Keep links and assets relative to their source file, within `docs/`. Use `.mdx` document links so the publishing step can preserve the selected product/version.
- No imports, exports, executable expressions, scripts, external includes, or secrets in synced MDX. Shared components are supplied by the docs application.
- Write answer-first summaries, descriptive headings, prerequisites, exact supported versions, reproducible steps, expected outcomes, and troubleshooting. Describe what instructional media shows in nearby text; do not invent transcripts or SEO claims.
- Main-branch docs are unreleased staging content. Production docs are captured from immutable release commits; do not create historical version folders or move release tags in this repository.
- The published site exposes version-aware Markdown, `llms.txt`, and a read-only MCP endpoint. Preserve factual product/version context and canonical links so agents can cite the correct release.
- Preview and validate through `poly-hammer-docs`: configure this checkout in `docs.local.json`, then run `npm run docs:prepare`, `npm run docs:validate`, and `npm run dev`. See that repository's README for the configuration schema.
- Update documentation with relevant feature changes and run the central documentation validator before submitting them.
