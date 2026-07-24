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

const renderer = await createCliRenderer({ exitOnCtrlC: true });
createRoot(renderer).render(<App />);
