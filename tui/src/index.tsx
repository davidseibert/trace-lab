/**
 * Entry point for the OpenTUI front-end.
 *
 * Run it in the container with an attached TTY:
 *
 *     make tui                       # brings the engine up first
 *     docker compose run --rm tui    # the raw equivalent
 */
import { createCliRenderer } from "@opentui/core";
import { createRoot } from "@opentui/react";

import { App } from "./App";
import { attachSpectate } from "./spectate";

const renderer = await createCliRenderer({ exitOnCtrlC: true });
attachSpectate(renderer); // read-only observer endpoint for the MCP bridge
createRoot(renderer).render(<App />);
