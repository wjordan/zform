# zform: notes for agents

## No backward compatibility

zform has no production users and no released format. Any part of the encoding may change in
place: wire layouts, node and payload kinds, lane operations, containers, and the CLI.

- Do not add version switches, compatibility readers or migration paths. Change the format
  together with its writer, reader, tests and docs.
- Delete a representation once no writer emits it. Do not keep a reader "for older files".
- Files packed by an earlier commit need not decode. Wire fixtures pin the current bytes;
  update them when the format changes on purpose.
- What still holds: a file round-trips byte-exact with the build that wrote it, and a format
  change has to pay for itself in size or speed without making the design sprawl.
