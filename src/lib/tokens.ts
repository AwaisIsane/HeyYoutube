// Token-budget helpers. All limits are read from the live session at runtime
// (session.contextWindow / session.contextUsage) — never hardcode the context window.

/** Tokens reserved for the model's response. */
export const OUTPUT_RESERVE = 1024;
/** Extra slack for prompt-template overhead and tokenizer variance. */
export const SAFETY_MARGIN = 128;

/**
 * Tokens still available for *new* input on this session, after accounting for
 * what's already in the context, the output reserve, and a safety margin.
 */
export function availableInput(
  session: LanguageModel,
  outputReserve = OUTPUT_RESERVE,
): number {
  //input quota is going to be depreceated 
  const left = ((session as any).contextWindow as number ?? session.inputQuota)  - ((session as any).contextUsage as number ?? session.inputUsage)
  return Math.max(0, left - outputReserve - SAFETY_MARGIN);
}
