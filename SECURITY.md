# Security

Report a vulnerability privately through the "Report a vulnerability" button on the repository's Security tab. Do not open a public issue. Include the deck, the command and `espice --version`.

Only the latest release is supported.

A plain netlist is data. If one crashes `espice`, that is a bug. If one alone leads to code execution, that is a vulnerability. A deck with `.hdl`, a VPI application or `.osdi` is code: `espice` compiles or loads it and runs it with your privileges, by design. Run such decks only from sources you trust.
